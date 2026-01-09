import asyncio  # For asynchronous operations
import logging  # For logging messages
import re  # For regular expressions
import json  # For JSON serialization/deserialization
import unicodedata
import hashlib # For hashing text for caching
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple  # For type annotations
from pymilvus import Collection, utility  # Milvus vector DB client
from API_Zilliz import ZillizVectorStorageService  # Custom wrapper for Zilliz/Milvus
from API_Azure import AzureOpenAIPlatformService  # Custom wrapper for Azure OpenAI
from config_public import ZILLIZ_URI, ZILLIZ_API_KEY  # Config values

class ZillizQueryOrchestrator:
    def __init__(self, uri=ZILLIZ_URI, api_key=ZILLIZ_API_KEY, db_alias: str = "default"):
        from pymilvus import connections  # Import connections for DB
        self.db_alias = db_alias  # Store DB alias
        connections.connect(alias=db_alias, uri=uri, token=api_key)  # Connect to Milvus/Zilliz
        self.client = ZillizVectorStorageService(uri=uri, api_key=api_key, db_name=db_alias)  # Init custom vector client
        self.azure = AzureOpenAIPlatformService()  # Init Azure OpenAI client
        self.cross_encoder = None  # Placeholder for cross-encoder (not used yet)
        self.embedding_cache = {} #Embedding cache
        self.max_cache_size = 1000  # Prevent unlimited growth

    # -------------------
    # AUTO-DETECT COLLECTIONS
    # -------------------
    # Mapping of collection names to their field names and keywords
    COLLECTION_FIELD_MAPPING = {
        "ProposalDocs": {
            "name_field": "file_name",
            "url_field": "sharepoint_link",
            "text_field": "chunk_text",
            "keywords": ["proposal", "contract", "bid"],
        },
        "MuniMagic": {
            "name_field": "file_name",
            "url_field": "sharepoint_link",
            "text_field": "chunk_text",
            "keywords": ["muni", "municipal", "magic", "regulation", "code", "standard" ],
        },
        "ProjectBriefs": {
            "name_field": "ShortName",
            "url_field": "ProjectURL",
            "text_field": "Brief",
            "keywords": ["project", "brief"],
        }
    }
    """ "Experiences": {
            "name_field": "ShortName",
            "url_field": "ExperienceURL",
            "text_field": "EmpDescription",
            "keywords": ["experience", "resume", "background"],
        },"""


    # Cached embedding generation
    def _get_cached_embedding(self, text: str) -> List[float]:
        """Get embedding from cache or generate new one."""
        # Create hash of the text for cache key
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Check cache: to look for existing embedding (similarity search optimization)
        if text_hash in self.embedding_cache:
            logging.info(f"Embedding cache HIT for query")
            return self.embedding_cache[text_hash]
        
        # Generate new embedding
        logging.info(f"Embedding cache MISS - generating new embedding")
        embedding = self.azure.generate_embedding(text)
        
        # Add to cache with size limit
        if len(self.embedding_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self.embedding_cache))
            del self.embedding_cache[first_key]
        
        self.embedding_cache[text_hash] = embedding
        return embedding

    def _detect_collection(self, conversation_context: str) -> List[str]:
        """
        Detect which collections to query based on conversation context.
        If it's a follow-up, the LLM will maintain topic continuity from context.
        """
        
        # Build collection list dynamically from active collections
        collection_descriptions = {
            "ProjectBriefs": "completed BGE projects, scope, team, costs, client info",
            "Experiences": "employee project history, roles, qualifications",
            "ProposalDocs": "RFPs, bids, SOQs, upcoming work, proposal teams",
            "MuniMagic": "codes, standards, regulations (IFC, IBC, MUTCD, ADA, etc.)"
        }
        
        collection_rules = {
            "ProjectBriefs": "project details, \"what did we do\", completed work",
            "Experiences": "staff qualifications, \"who worked on\", employee roles",
            "ProposalDocs": "proposals, bids, RFPs, upcoming projects",
            "MuniMagic": "code requirements, standards, regulations"
        }
        
        # Build prompt sections only for collections in COLLECTION_FIELD_MAPPING
        collections_text = "\n".join(
            f"        {i+1}. {coll} - {collection_descriptions[coll]}"
            for i, coll in enumerate(self.COLLECTION_FIELD_MAPPING.keys())
            if coll in collection_descriptions
        )
        
        rules_text = "\n".join(
            f"        - Use {coll} for: {collection_rules[coll]}"
            for coll in self.COLLECTION_FIELD_MAPPING.keys()
            if coll in collection_rules
        )

        message = [
            {
                "role": "system",
                "content": f"""Route questions to one or more collections:
                    Collections:
                    {collections_text}

                    Collection description:
                    {rules_text}
                    
                    - Review the conversation history to understand the topic flow
                    - Determine which collection(s) best match the current question
                    - If the keywords indicate a specific collection, choose that one
                    - If the conversation is continuing the same topic, maintain context continuity
                    - If the question shifts to a completely new topic, route based on the new topic
                    - Include multiple collections if question spans categories

                    Return ONLY a JSON array of collection names, nothing else.
                    Example: ["ProjectBriefs","ProposalDocs"]"""     
            },
            {
                "role": "user",
                "content": conversation_context,
            }
        ]

        # Ask LLM to determine collection name
        raw_answer = self.azure.chat_completion(message)

        try:
            answer = json.loads(raw_answer)
            if not isinstance(answer, list):
                raise ValueError("LLM did not return a list.")
        except Exception as e:
            logging.error(f"Failed to parse collection list: {e}. Raw: {raw_answer}")
            # Fallback: try to extract collection names from response
            answer = []
            for coll in self.COLLECTION_FIELD_MAPPING.keys():
                if coll in raw_answer:
                    answer.append(coll)
            if not isinstance(answer, list):
                raise ValueError("LLM did not return a list.")
            
        logging.info(f"Auto-detected collection answer: {answer}")
        return answer
    
    def _detect_person_targets_llm(self, conversation_context: str) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": "Extract all person names from the conversation. Include names from both the current question AND recent context. Return ONLY a JSON array of names. Example: [\"John Smith\",\"Jane Doe\"]"
            },
            {
                "role": "user",
                "content": conversation_context  # Pass full context instead of just question
            }
        ]
        
        try:
            raw = self.azure.chat_completion(messages)
            names = json.loads(raw)
            if isinstance(names, list):
                return [n.strip() for n in names if n.strip()]
        except Exception as e:
            logging.warning(f"LLM name extraction failed: {e}")
        
        return []
    
    # -------------------
    # MAIN ORCHESTRATOR
    # -------------------
    async def orchestrate_query(
        self,
        question: str,
        raw_history: List[Dict[str, str]] = None,
        collection_name: Optional[str] = None,  
        global_top_k: int = 8,
        per_collection_k: int = 20,
        filter_expr: Optional[str] = None,
        enforce_diversity: bool = False,  # now optional
    ) -> Dict[str, Any]:
        
        query_embedding = self._get_cached_embedding(question)  # Generate a vector embedding for the user's question using the Azure OpenAI service
        
        # Reformat history from json to flat list
        history = []
        if raw_history and len(raw_history) > 0:
            h0 = raw_history[0]
            summary = h0.get("olderResponseSummary")
            recent = h0.get("recentResponses") or []

            if summary:
                history.append({"role": "system", "content": summary})

            for m in recent:
                role = m.get("role")
                content = m.get("content")
                if role and content:
                    history.append({"role": role, "content": content})

        conversation_context = ""   
        if history and len(history) > 0:
            conversation_context = "Conversation history:\n" + "\n".join(
                f"{m['role']}: {m['content'][:300]}"  # Truncate each message to 300 chars
                for m in history
            ) + f"\n\nCurrent question: {question}"
        else:
            conversation_context = f"Current question: {question}"

        # Run collection detection and name detection in parallel
        if collection_name:
            collection_name = [collection_name]  # Use provided collection
            logging.info(f"Using provided collection: {collection_name}")
            print(f"Using provided collection: {collection_name}")
        else:
            collection_name = await asyncio.to_thread(self._detect_collection, conversation_context)
            logging.info(f"Auto-detected collections: {collection_name}")
        
        # Start name detection in parallel (always runs)
        name_task = asyncio.create_task(
            asyncio.to_thread(self._detect_person_targets_llm, conversation_context)
        )

        # Smart per-collection sizing
        num_collections = len(collection_name)
        if num_collections == 1:
            adjusted_per_collection_k = 15
        elif num_collections == 2:
            adjusted_per_collection_k = 10
        else:
            adjusted_per_collection_k = 8
        logging.info(f"Adjusted per_collection_k to {adjusted_per_collection_k} for {num_collections} collections")

        # Step 1: Collect results
        if not collection_name:
            raise ValueError("Error: No collection detected for the question.")
        # Create all search tasks
        search_tasks = [
            self._search_one_collection(c_name, query_embedding, adjusted_per_collection_k, filter_expr)
            for c_name in collection_name
        ]
        
        # Execute all searches in parallel
        logging.info(f"Starting parallel search across {len(search_tasks)} collections")
        search_results = await asyncio.gather(*search_tasks)
        
        # Flatten results
        results = []
        for result_list in search_results:
            results.extend(result_list)
        
        logging.info(f"Parallel search completed: {len(results)} total results")
        
        # Wait for name detection to complete (started earlier in parallel)
        targets = await name_task
        logging.info(f"Detected name targets: {targets}")

        # Build normalized variants and create name-based filter
        aug_result = []
        all_variants = []
        if targets:
            all_variants = sorted(
                {v for t in targets for v in self._generate_name_variants(t)}
            )
        
            # Over-fetch without DB-side filter
            fetch_k = max(adjusted_per_collection_k, 100)

            # Parallel fetch for name-based queries
            name_search_tasks = [
                self._search_one_collection(c_name, query_embedding, fetch_k, filter_expr=None)
                for c_name in collection_name
            ]
            name_search_results = await asyncio.gather(*name_search_tasks)
            
            for result_list in name_search_results:
                aug_result.extend(result_list)
                
            # Keep only hits whose extracted text contains any name variant
            def _has_literal(txt: str) -> bool:
                T = self._normalize(txt or "")
                return any(self._normalize(v) in T for v in all_variants)
            aug_result = [r for r in aug_result if _has_literal(self._extract_text(r))]

        # merge with base results before building candidates
        results = (results or []) + aug_result

        # de-dup by (collection,id)
        seen = set()
        uniq = []
        for r in results:
            key = (r["collection"], r["id"])
            if key not in seen:
                uniq.append(r); seen.add(key)
        results = uniq

        # Step 2: Build candidates for reranking
        logging.info(f"Begin candidate building from {len(results)} results")
        candidates = []
        for r in results:
            text = self._extract_text(r)  # Extract text from result
            candidates.append((question, text, r))  # Tuple for reranking

        # Apply literal boost (precision) before reranking
        candidates = self._apply_literal_gate(candidates, all_variants)

        # Skip LLM reranking if results are high quality
        if len(candidates) <= 10 and all(c[2]['distance'] > 0.75 for c in candidates[:10]):
            # High confidence - skip LLM reranking, use vector similarity
            logging.info("High confidence results - skipping LLM reranking")
            reranked = sorted(candidates, key=lambda x: x[2]['distance'], reverse=True)
            
            # Assign scores based on distance
            for idx, (_, _, r) in enumerate(reranked):
                r["rerank_score"] = r['distance']  # Use vector distance as score
        else:
            # Step 3: LLM reranking
            logging.info(f"Begin LLM reranking of {len(candidates)} candidates")
            # Truncate snippets to reduce token usage
            snippets = "\n\n".join(
                f"[{i+1}] {text[:300]}" for i, (_, text, _) in enumerate(candidates)
            )

            messages = [
                {"role": "system", "content": "You are a reranker. Rank snippets by relevance. Return a comma-separated list of numbers."},
                {"role": "user", "content": f"Query: {question}\n\nSnippets:\n{snippets}\n\nBest order (e.g., '2,1,3'):"}
            ]

            ranking_answer = self.azure.chat_completion(messages)  # Get reranking order from LLM
            logging.info(f"LLM reranker answer: {ranking_answer}")

            # Parse order like "2,1,3..."
            order = self._safe_parse_reranker(ranking_answer, len(candidates))
            reranked = [candidates[i] for i in order if 0 <= i < len(candidates)]  # Reorder candidates

            # assign scores
            for idx, (_, _, r) in enumerate(reranked):
                r["rerank_score"] = float(len(reranked) - idx)  # Higher rank = higher score

        results = [r for (_, _, r) in reranked]  # Takes the reranked list, which contains tuples of (question, text, result_dict), and extracts only the result_dict part from each tuple

        # Step 4: Deduplication logic (Not used)
        logging.info("Begin deduplication")
        # Take top K results by relevance score, regardless of collection
        if enforce_diversity:
            seen = set()
            deduped = []
            for r in results:
                if r["collection"] not in seen:
                    deduped.append(r)
                    seen.add(r["collection"])
                    # global_top_k = 8
                if len(deduped) >= global_top_k:
                    break
        else:
            deduped = results[:global_top_k]  # Just take top K

        # Step 5: Build context for answer generation
        logging.info(f"Begin context building from {len(deduped)} documents")
        context_parts = []
        for i, r in enumerate(deduped, start=1):
            r["chunk_number"] = i
            context_parts.append(
                f"[{i}] From {r['collection']} (score={r['rerank_score']:.3f}):\n{self._extract_text(r)}"
            )
        context = "\n\n".join(context_parts)

        # Early exit if empty context
        if not context.strip():
            return {"answer": "I wasn't able to find relevant information in the retrieved documents.",
                    "references": [], "chunks": deduped}

        # Strong rule in the prompt so LLM never chooses fallback when names match ----
        target_entities_str = ", ".join(f'"{t}"' for t in targets) if targets else "[]"
        strict_name_rule = (
            "Relevance rule: If any snippet contains an exact, case-insensitive match to "
            "any target entity or its normalized variants, you MUST include those documents "
            "and MUST NOT return the fallback.\n"
            f"Target entities: [{target_entities_str}]\n"
        )

        # Step 6: Generate final answer
        # Enhanced system prompt with stricter citation requirements
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful assistant that outputs search results in a professional, project-index style.\n\n"
                    "## CRITICAL RULES\n" 
                    f"{strict_name_rule}"
                    "- Each context snippet is labeled like [1], [2], etc."
                    "- When an item in your answer is supported by a snippet, include the corresponding [n] from that snippet in your answer."
                    "- Keep the same [n] labels as shown in the context snippets. Do not renumber them."
                    "- Use ONLY the provided context snippets to answer.\n"
                    "- Combine and summarize information across snippets for each document. You must provide a reference to each corresponding snippet using the [n] labels.\n"
                    "- Each document should be shown as one entry in a **file navigation list** (filenavlist).\n"
                    "- For each file, provide:\n"
                    "  • **Document title** (bolded)\n"
                    "  • A concise, factual summary (2–4 sentences) describing the project, scope, location, and context.\n"
                    "  • Mention key disciplines, clients, and any employee(s) relevant to the query.\n"
                    "- Provide a confidence level (0-100) for each piece of information provided.\n"
                    "- If confidence level < 70, do not include that information.\n"
                    "- If no relevant information is found, do not return any reference. Instead, respond with a reasonable sentence, such as \"I wasn't able to find relevant information in the retrieved documents.\".\n"
                    "- Output must begin with a filenavlist. Do not output the word \"filenavlist\".\n"
                    "- Use the following syntax:\n"
                    "  \n"
                    "- Each entry should contain only one concise description paragraph.\n"
                    "- Do NOT include headers like '## Results' or '## Files' – start directly with the navlist.\n"
                    "- If question involves multiple projects or proposals, separate each project/proposal with a new line in the output.\n"
                    "- If the question involves two or more sections, such as involving both project briefs and proposals, separate the sections clearly with reasonable starting sentences.\n"
                    "- DO NOT INCLUDE CONFIDENCE LEVEL IN THE OUTPUT. THIS IS FOR YOUR INTERNAL USE ONLY.\n"
                    "- Ensure all information is accurate and based solely on the provided snippets.\n"

                    "## Follow-up Question Handling\n"
                    "- If this is a follow-up question (e.g., 'provide references', 'tell me more'), use the conversation context to understand what the user is referring to\n"
                    "- Make sure your answer is consistent with the previous conversation\n"
                    "- Always include citations even for follow-up questions\n"
                ),
            },
            {
                "role": "user",
                "content": f"{conversation_context}\n\nCurrent question: {question}\n\nContext snippets:\n{context}",
            },
        ]

        answer = self.azure.chat_completion(messages)  # Generate answer using LLM

        sources = []
        for r in deduped:
            meta = r.get("metadata", {})
            entity = meta.get("entity", {})
            src = {"source_id": r["chunk_number"], **meta, **entity}
            sources.append(src)  # Build sources for references

        answer_clean, reference_list = self.process_answer(answer, sources, deduped)  # Add references
        
        return {
            "answer": answer_clean,
            "references": reference_list,
            "chunks": deduped
        }

    # -------------------
    # SEARCH HELPERS
    # -------------------
    # This function performs a vector similarity search on a single Milvus/Zilliz collection.
    async def _search_one_collection(
        self, collection_name: str, query_embedding: List[float], top_k: int, filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        
        coll = Collection(collection_name, using=self.db_alias)
        # 1) Get actual non-vector fields from schema
        schema_fields = [f.name for f in coll.schema.fields if f.dtype.name != "FLOAT_VECTOR"]
        schema_field_set = set(schema_fields)

        # 2) Decide which optional carriers to request (only if they truly exist)
        want_common = {"chunk_text", "file_name", "sharepoint_link"}
        want_json_like = {"dynamic_field"}  # if you created a JSON column with this exact name (underscore)

        fields = set(schema_fields)  # start with known schema fields

        # request JSON column only if it's in schema
        fields |= (want_json_like & schema_field_set)
        # request common fallbacks only if they exist in schema
        fields |= (want_common & schema_field_set)

        # include $meta only when dynamic field is enabled for this collection
        if getattr(coll.schema, "enable_dynamic_field", False):
            fields.add("$meta")

        fields = list(fields)

        hits = coll.search(
            data=[query_embedding],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            filter=filter_expr,
            output_fields=fields,
        )

        results = []
        for hit in hits[0]:
            meta = dict(hit.entity)

            # 3) Merge dynamic carrier(s)
            if "$meta" in meta:
                try:
                    meta.update(json.loads(meta["$meta"]))
                except Exception:
                    pass
                meta.pop("$meta", None)

            if "dynamic_field" in meta:
                df_val = meta.get("dynamic_field")
                if isinstance(df_val, str):
                    try:
                        meta.update(json.loads(df_val))
                    except Exception:
                        # keep as raw blob so downstream extract can see it
                        meta.setdefault("chunk_text", df_val)
                elif isinstance(df_val, dict):
                    meta.update(df_val)

            # SIMPLIFIED: Extract text once instead of duplicate logic
            result_dict = {
                "collection": collection_name,
                "id": hit.id,
                "distance": hit.distance,
                "metadata": meta,
            }
            text = self._extract_text(result_dict)
            result_dict["chunk_text"] = text
            
            employees = self._extract_employees_from_text(text)
            if employees:
                meta["employees"] = employees

            results.append(result_dict)
        return results

    # ========== NAME AWARENESS HELPERS ==========
    def _normalize(self, s: str) -> str:
        """Lowercase, strip accents/extra spaces, collapse punctuation."""
        if not isinstance(s, str):
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = s.lower().strip()
        # collapse multiple spaces
        s = re.sub(r"\s+", " ", s)
        return s

    def _tokenize_name(self, name: str) -> List[str]:
        name = self._normalize(name)
        name = re.sub(r"[^a-z0-9\s\.\-']", " ", name)  # keep dots for initials like "c."
        return [t for t in name.split() if t]

    def _nickname_map(self) -> Dict[str, List[str]]:
        # Minimal set; expand as needed
        return {
            "robert": ["rob", "bob", "bobby", "robby"],
            "william": ["bill", "billy", "will"],
            "richard": ["rick", "ricky", "rich", "dick"],
            "michael": ["mike", "mikey"],
            "christopher": ["chris", "kit"],
            "andrew": ["andy", "drew"],
            "james": ["jim", "jimmy"],
            "jose": ["pepe"],  # accentless handled by _normalize
            "elizabeth": ["liz", "lizzy", "beth", "eliza"],
            "katherine": ["kat", "kate", "katie", "catherine"],
            "margaret": ["meg", "maggie", "peggy"],
            # add more over time
        }

    def _generate_name_variants(self, full_name: str) -> List[str]:
        """
        From 'Christopher A. Brown', generate:
        - chris brown, christopher brown, c. brown, chris b, christopher b
        - brown, chris; brown, christopher
        - diacritic-free by normalization
        """
        toks = self._tokenize_name(full_name)
        if not toks:
            return []

        # Separate likely given/middle/surname (heuristic)
        # Keep last token as surname if >=2 tokens
        given = toks[0]
        middle = toks[1:-1] if len(toks) > 2 else []
        surname = toks[-1] if len(toks) >= 2 else ""

        variants = set()

        # base forms
        base = " ".join(toks)
        variants.add(base)

        # nickname expansions on given name
        nn = self._nickname_map().get(given, []) + [given]

        # initials
        given_initial = f"{given[0]}." if given else ""
        surname_initial = f"{surname[0]}." if surname else ""

        # build variants
        for g in nn:
            if surname:
                variants.add(f"{g} {surname}")
                variants.add(f"{surname}, {g}")
                variants.add(f"{g} {surname_initial}".strip())
                variants.add(f"{given_initial} {surname}".strip())
                variants.add(f"{given_initial} {surname_initial}".strip())
                variants.add(f"{g[0]}. {surname}".strip())
                variants.add(f"{g} {surname[0]}".strip())
            else:
                variants.add(g)

        # add forms with middle initials if present
        if surname and middle:
            m_i = " ".join([f"{m[0]}." for m in middle if m])
            for g in nn:
                variants.add(f"{g} {m_i} {surname}".strip())
                variants.add(f"{surname}, {g} {m_i}".strip())

        # collapse spaces and ensure deterministic
        variants = {self._normalize(v) for v in variants if v.strip()}
        return sorted(variants)


    def _has_any_name(self, text: str, all_variants: List[str]) -> bool:
        T = self._normalize(text)
        return any(v in T for v in all_variants if v)
    
    # Improves precision for name-based queries. It boosts and reorders candidates based on literal name matches before sending them to the LLM reranker.
    def _apply_literal_gate(self, candidates: List[Tuple[str, str, Dict[str, Any]]], variants: List[str], top_k: int = 50):
        """
        Boost literal matches as a precision helper. Return re-ordered candidate list.
        """
        def has_name(txt: str) -> bool:
            t = self._normalize(txt)
            return any(re.search(rf"\b{re.escape(v)}\b", t) for v in variants)

        boosted = []
        for q, t, r in candidates:
            bonus = 0.7 if variants and has_name(t) else 0.0
            # lower distance is better; subtract bonus to make literal hits rank higher
            effective_distance = r.get("distance", 0.0) - bonus
            boosted.append((q, t, {**r, "boost": bonus, "effective_distance": effective_distance}))

        boosted.sort(key=lambda x: x[2].get("effective_distance", x[2].get("distance", 0.0)))
        return boosted[:top_k]

    def _safe_parse_reranker(self, ranking_answer: str, n: int) -> List[int]:
        """
        Parse '2,1,3' to [1,0,2]. If empty/garbled, return identity order.
        """
        try:
            order = [int(x.strip())-1 for x in ranking_answer.split(",") if x.strip().isdigit()]
            if not order:
                return list(range(n))
            order = [i for i in order if 0 <= i < n]
            return order or list(range(n))
        except Exception:
            return list(range(n))
    
    # -------------------
    # UTILITIES
    # -------------------
    def process_answer(self, answer: str, sources: List[Dict[str, Any]], results: List[Dict[str, Any]]):
        """Extract references from answer and build reference list with collection names."""
        reference_list = []
        used_refs = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))  # Find all citations
        
        # If no citations found (e.g., fallback answer), return empty reference list
        if not used_refs:
            logging.info("No citations found in answer - returning empty reference list")
            return answer, reference_list
        
        result_lookup = {r["chunk_number"]: r for r in results}  # Map chunk number to result

        # Build reference list and URL mapping
        url_map = {}
        for ref_id in sorted(used_refs):  # For consistent ordering
            chunk = result_lookup.get(ref_id)
            if not chunk:
                continue
            
            collection = chunk.get("collection", "ProposalDocs")
            mapping = self.COLLECTION_FIELD_MAPPING.get(collection, {})
            name_field, url_field = mapping.get("name_field", "file_name"), mapping.get("url_field", "ProjectURL")

            meta, entity = chunk.get("metadata", {}), chunk.get("metadata", {}).get("entity", {})
            file_name = (
                entity.get(name_field)
                or entity.get("LongName")
                or entity.get("ShortName")
                or meta.get(name_field)
                or "Unknown Document"
            )
            url = entity.get(url_field) or meta.get(url_field) or "#"

            reference_list.append({
                "id": ref_id,
                "collection": collection,
                "fileName": file_name,
                "fileUrl": url,
            })
            
            # Store URL for replacement
            url_map[ref_id] = url

        # Replace citation markers with clickable Markdown links
        def replace_citation(match):
            ref_id = int(match.group(1))
            url = url_map.get(ref_id, "#")
            return f"[[{ref_id}]]({url})"
        
        answer = re.sub(r"\[(\d+)\]", replace_citation, answer)

        return answer, reference_list

    def _extract_employees_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract employee information from chunk text."""
        try:
            # Look for the Employees section with JSON data
            if match := re.search(r'Employees:\s*(\[[\s\S]*?\])', text):
                employees_json = match.group(1)
                return json.loads(employees_json)
        except Exception as e:
            logging.warning(f"Failed to parse employee data: {e}")
        return []

    def _extract_text(self, result: Dict[str, Any]) -> str:
        """Extract full text content from the result, including metadata."""
        mapping = self.COLLECTION_FIELD_MAPPING.get(result["collection"], {})
        text_field = mapping.get("text_field", "chunk_text")
        meta = result["metadata"]
        entity = meta.get("entity", {})

        # SIMPLIFIED: Consolidated text extraction logic
        text = (
            meta.get("chunk_text") or
            entity.get(text_field) or
            meta.get(text_field) or
            ""
        )
        
        # Fallback: combine all string values
        if not text:
            combined = list(meta.values()) + list(entity.values())
            text = " ".join(str(v) for v in combined if isinstance(v, str))
        
        return text  # Return full text without truncation
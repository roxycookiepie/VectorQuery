import logging  # Standard Python logging module for logging messages
import json  # Standard Python module for working with JSON data
import azure.functions as func  # Azure Functions Python SDK for HTTP and trigger bindings
import azure.durable_functions as df  # Azure Durable Functions SDK for orchestrations and activities
from vector_sync.vector_sync_orchestrator import VectorSyncOrchestrator  # Custom orchestrator for vector sync operations
from zilliz_query_all import ZillizQueryOrchestrator  # Custom orchestrator for querying Zilliz vector DB
from generate_proposal import ProposalGenerator  # Custom class for proposal generation from PDF
import base64

# Create a FunctionApp instance with anonymous HTTP authentication
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# ---------------------------
# Vector Query Endpoint
# ---------------------------
@app.route(route="vector-query", methods=["POST"])  # HTTP POST endpoint for querying vectors
async def vector_query_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()  # Parse JSON body from request;
        logging.info(f"Received vector-query request (keys): {list(req_body.keys())}")
    except ValueError:
        logging.warning("Invalid JSON body received in vector-query endpoint.")
        return func.HttpResponse("Invalid JSON body.", status_code=400)
    raw_history = req_body.get("history", [])  # Get history from request, default to empty list
    question = req_body.get("question")  # Get question from request
    collection_name = req_body.get("collection_name")  # Get collection name from request

    logging.info(f"vector-query parameters: question={question}")

    if not question:  # Validate required parameter
        logging.warning("Missing required parameter: question in vector-query endpoint.")
        return func.HttpResponse("Missing required parameter: question.", status_code=400)

    try:
        logging.info("About to create ZillizQueryOrchestrator")
        orchestrator = ZillizQueryOrchestrator()  # Create orchestrator instance
        logging.info("Successfully created orchestrator")

        answer = await orchestrator.orchestrate_query(
            question=question,
            raw_history=raw_history,
            collection_name=collection_name
        )  # Run the query
        logging.info(f"vector-query response: {answer}")
        return func.HttpResponse(json.dumps({"answer": answer}), mimetype="application/json")
    except Exception as e:
        logging.exception("Error in vector query endpoint")  # Log exception
        return func.HttpResponse(json.dumps({"status": "error", "message": str(e)}), status_code=500, mimetype="application/json")


# ---------------------------
# Generate Proposal Endpoint
# ---------------------------
@app.route(route="generate-proposal", methods=["POST"])
async def generate_proposal_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info("Received generate-proposal request.")

        pdf_content = None
        additional_context = None

        # Try JSON first
        try:
            body = req.get_json()
            b64 = body.get("file_base64")
            additional_context = body.get("context")
            if b64:
                pdf_content = base64.b64decode(b64)
        except Exception:
            pass

        # Fall back to multipart if JSON not provided
        if pdf_content is None:
            files = req.files.getlist('file')
            if not files:
                return func.HttpResponse("No PDF uploaded.", status_code=400)
            pdf_file = files[0]
            pdf_content = pdf_file.read()
            additional_context = req.form.get('context', None)

        generator = ProposalGenerator()
        result = generator.process_pdf_and_generate(pdf_content, additional_context)

        if result.get("status") == "success":
            return func.HttpResponse(
                body=result["pdf_bytes"],
                mimetype="application/pdf",
                headers={"Content-Disposition": "inline; filename=generated_proposal.pdf"}
            )

        return func.HttpResponse(
            json.dumps({"status": "error", "message": result.get("message")}),
            status_code=500,
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("Error in generate proposal endpoint")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

# ---------------------------
# Delete Vectors Endpoint
# ---------------------------
@app.route(route="vector-delete", methods=["POST"])  # HTTP POST endpoint for deleting vectors
def vector_delete_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()  # Parse JSON body from request
    except ValueError:
        return func.HttpResponse("Invalid JSON body.", status_code=400)  # Return error if JSON is invalid

    collection_name = req_body.get("collection_name")  # Get collection name from request
    sharepoint_file_id = req_body.get("sharepoint_file_id")  # Get SharePoint file ID from request

    if not collection_name or not sharepoint_file_id:  # Validate required parameters
        return func.HttpResponse(
            "Missing required parameters: collection_name and sharepoint_file_id.", status_code=400
        )

    orchestrator = VectorSyncOrchestrator()  # Create orchestrator instance
    try:
        orchestrator.delete_file_by_sharepoint_id(collection_name, sharepoint_file_id)  # Delete vectors by file ID
        msg = f"Deleted all vectors for SharePoint file ID {sharepoint_file_id} in collection {collection_name}."
        return func.HttpResponse(json.dumps({"status": "success", "message": msg}), mimetype="application/json")
    except Exception as e:
        logging.exception("Error in vector delete endpoint")  # Log exception
        return func.HttpResponse(json.dumps({"status": "error", "message": str(e)}), status_code=500, mimetype="application/json")


# ---------------------------
# Vector Sync Single Endpoint: For project proposals and Munimagic files
# ---------------------------
@app.route(route="vector-sync-single", methods=["POST"])  # HTTP POST endpoint for syncing a single file
async def vector_sync_single_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()  # Parse JSON body from request
    except ValueError:
        return func.HttpResponse("Invalid JSON body.", status_code=400)

    collection_name = req_body.get("collection_name")  # Get collection name from request
    file_url = req_body.get("url")  # Get file URL from request

    if not collection_name or not file_url:  # Validate required parameters
        return func.HttpResponse("Missing required parameters: collection_name and url.", status_code=400)

    try:
        orchestrator = VectorSyncOrchestrator()  # Create orchestrator instance
        orchestrator.upsert_single_file_from_sharepoint_url(collection_name, file_url)  # Sync single file
        msg = f"Single file import completed for {file_url}."
        return func.HttpResponse(json.dumps({"status": "success", "message": msg}), mimetype="application/json")
    except Exception as e:
        logging.exception("Error in vector sync single endpoint")  # Log exception
        return func.HttpResponse(json.dumps({"status": "error", "message": str(e)}), status_code=500, mimetype="application/json")


# ---------------------------
# Dynamics Sync Endpoint: For SQL DB: employee experiece & project briefs
# ---------------------------
@app.route(route="Dynamics_Sync", methods=["POST"])  # HTTP POST endpoint for Dynamics SQL-to-vector sync
def dynamics_sync_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    from vector_sync.Dynamics_sync import DynamicsSync  # Import inside function for isolation
    try:
        logging.info("Starting Dynamics SQL-to-vector sync.")
        sync = DynamicsSync()  # Create DynamicsSync instance
        sync.process_all_queries()  # Process all queries
        logging.info("Dynamics SQL-to-vector sync completed successfully.")
        return func.HttpResponse(
            json.dumps({"status": "success", "message": "Dynamics SQL-to-vector sync completed successfully."}),
            mimetype="application/json"
        )
    except Exception as e:
        logging.exception("Error during Dynamics SQL-to-vector sync")  # Log exception
        return func.HttpResponse(json.dumps({"status": "error", "message": str(e)}), status_code=500, mimetype="application/json")
    

# ---------------------------
# Durable Functions: for bulk upload from SharePoint folder
# ---------------------------

# Starter - for document_parsing_and_chunking_service
@app.route(route="vector-sync-bulk", methods=["POST"])  # HTTP POST endpoint to start bulk upload
@app.durable_client_input("starter")  # Durable Functions client input binding
async def durable_bulk_upload_start(req: func.HttpRequest, starter: df.DurableOrchestrationClient) -> func.HttpResponse:
    client = starter  # Durable orchestration client

    try:
        body = req.get_json()  # Parse JSON body from request
    except ValueError:
        return func.HttpResponse("Invalid JSON body.", status_code=400)

    collection_name = body.get("collection_name")  # Get collection name from request
    folder_url = body.get("url")  # Get folder URL from request

    if not collection_name or not folder_url:  # Validate required parameters
        return func.HttpResponse("Missing required parameters: collection_name and url", status_code=400)

    instance_id = await client.start_new(
        "durable_bulk_upload_orchestrator",  # Name of orchestrator function
        None,
        {"collection_name": collection_name, "folder_url": folder_url},  # Input payload
    )

    logging.info(f"🚀 Started Durable bulk upload with instance ID {instance_id}")
    return client.create_check_status_response(req, instance_id)  # Return status response

@app.activity_trigger(input_name="payload")
def durable_list_files_activity(payload):
    folder_url = payload["folder_url"]

    orchestrator = VectorSyncOrchestrator()
    files = [
        sp_file.additional_graph_fields.get("webUrl") or folder_url
        for sp_file in orchestrator._sp.sharepoint_iter_file_descriptors_under_url(
            folder_url,
            allowed_extensions=(".pdf", ".docx", ".txt")
        )
    ]

    # Make ordering deterministic
    files.sort()
    return files

# Orchestrator - document_parsing_and_chunking_service
@app.orchestration_trigger(context_name="context")  # Durable Functions orchestration trigger
def durable_bulk_upload_orchestrator(context: df.DurableOrchestrationContext):
    try:
        payload = context.get_input()
        collection_name = payload["collection_name"]
        folder_url = payload["folder_url"]

        # Deterministic: result is recorded in history
        files = yield context.call_activity(
            "durable_list_files_activity",
            {"folder_url": folder_url}
        )

        results = []
        batch_size = 1

        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            tasks = [
                context.call_activity("durable_bulk_upload_activity", (collection_name, f))
                for f in batch
            ]
            batch_results = yield context.task_all(tasks)
            results.extend(batch_results)

        return results
    except Exception as exc:
        logging.exception(f"Error in durable_bulk_upload_orchestrator: {exc}")
        return [{"status": "error", "message": str(exc)}]


# Activity - document_parsing_and_chunking_service
@app.activity_trigger(input_name="context")  # Durable Functions activity trigger
def durable_bulk_upload_activity(context):
    try:
        collection_name, file_url = context  # Unpack context tuple
        orchestrator = VectorSyncOrchestrator()  # Create orchestrator instance
        orchestrator.upsert_single_file_from_sharepoint_url(collection_name, file_url)  # Sync single file
        logging.info(f"✅ Processed {file_url}")
        return {"file_url": file_url, "status": "success"}  # Return success result
    except Exception as exc:
        logging.exception(f"Error processing file {file_url if 'file_url' in locals() else None}: {exc}")
        return {"file_url": file_url if 'file_url' in locals() else None, "status": "error", "message": str(exc)}
    

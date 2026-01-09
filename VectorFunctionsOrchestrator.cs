using Microsoft.Graph.Models;
using Microsoft.Identity.Client;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web.Helpers;
using System.Web.UI.WebControls;
using Technology_Solutions.Models;
using TechnologySolutions;

namespace Technology_Solutions
{
    public class VectorFunctionsOrchestrator
    {
        private readonly HttpClient _httpClient;
        private readonly LogFile _logFile;
        private readonly string blobStorageConnectionString = System.Configuration.ConfigurationManager.AppSettings["blobStorageConnectionString"];

        // Add these fields to your class (or get from config)
        private readonly string tenantId = System.Configuration.ConfigurationManager.AppSettings["azureTenantId"];
        private readonly string clientId = System.Configuration.ConfigurationManager.AppSettings["AzureClientId"];
        private readonly string clientSecret = System.Configuration.ConfigurationManager.AppSettings["AzureClientSecret"];
        private readonly string functionAppResource = System.Configuration.ConfigurationManager.AppSettings["FunctionAppResource"]; // e.g., api://<app-id-guid>


        public VectorFunctionsOrchestrator(HttpClient httpClient, LogFile logFile)
        {
            _httpClient = httpClient;
            _httpClient.Timeout = TimeSpan.FromMinutes(5);
            _logFile = logFile;
        }

        public async Task<OpenAIChatResponse> HandleQuestionAsync(QuestionRequest request)
        {
            _logFile.MyLogFile("INFO", $"HandleQuestionAsync called. Question: {request.question}");
            var azureOpenAI = new AzureOpenAIAPI(_httpClient, _logFile);

            // Fetch conversation history from Blob
            List<ConversationHistory> history = new List<ConversationHistory>();
            Azure.Storage.Blobs.BlobContainerClient containerClient = null;
            string blobPath = null;
            if (!string.IsNullOrWhiteSpace(request.blobName))
            {
                try
                {
                    var blobServiceClient = new Azure.Storage.Blobs.BlobServiceClient(blobStorageConnectionString);
                    containerClient = blobServiceClient.GetBlobContainerClient("ai-agent-history");
                    blobPath = request.blobName.StartsWith("history/") ? request.blobName : $"history/{request.blobName}";
                    var blobJson = await GetBlobJsonAsync(blobPath, containerClient);
                    history = JsonConvert.DeserializeObject<List<ConversationHistory>>(blobJson);
                }
                catch (Exception ex)
                {
                    _logFile.MyLogFile("ERROR", $"Error fetching conversation history: {ex.Message}");
                }
            }

            var orchestratorPrompt = new List<OpenAIChatMessage>
            {
                new OpenAIChatMessage
                {
                    role = "system",
                    content = @"You are a routing assistant.
                    Your goal is to determine which workflow a user’s question should be routed to, and how confident you are in that decision.

                    Available workflows:
                    1. QueryVectorDB — for questions answerable using documents in the vector database.
                        Includes:
                        - BGE Project Briefs (overview, highlights, timelines, costs, employees, contacts)
                        - BGE Project Employee Experience (employee roles and experiences)
                        - BGE Project Proposals (initial proposals sent to clients)
                        - Municipality Guidelines (county, city, state, international)

                    2. GenerateProposal — for requests to generate a similar proposal based on a user-uploaded document (PDF or word document).

                    ---
                    HARD RULE:
                    - Only select 'GenerateProposal' if the user has uploaded a file (i.e., 'uploadedFileBlobName' exists or 'FileUploaded: true' is present).
                    - If no file is uploaded, do NOT select 'GenerateProposal' under any circumstances.

                    When deciding:
                    - Think carefully about which workflow best matches the user’s question.

                    ---

                    **Response Rules:**
                    1. If confidence >= 70: RETURN ONLY THE WORKFLOW NAME.
                        - Return:
                        [Workflow Name]                            

                    2. If confidence < 70 but the question could fit one of the workflows:
                        - Ask a clarifying question:
                            FollowUp: [Your clarifying question here]

                    3. If the question clearly does not fit any defined workflow:
                        - Return:
                            None

                    **Example Outputs:**
                    - QueryVectorDB

                    - FollowUp: Can you clarify which project or municipality this question is about?

                    - None
                "
                },
                new OpenAIChatMessage
                {
                    role = "system",
                    content = $"Conversation history:{JsonConvert.SerializeObject(history)}"
                },
                new OpenAIChatMessage
                {
                    role = "user",
                    content = request.question
                }
            };

            // Ask LLM which agent to use
            var agent = await azureOpenAI.CallAzureOpenAI(orchestratorPrompt);
            _logFile.MyLogFile("INFO", $"Agent selected: {agent}");

            string finalAnswer = null;
            List<References> finalReferences = null;

            // Route to the selected agent
            switch (true)
            {
                case var _ when agent.StartsWith("FollowUp:"):
                    finalAnswer = agent.Replace("FollowUp:", "").Trim();
                    break;

                case var _ when agent.Trim() == "QueryVectorDB":
                    try
                    {
                        var vectorResponse = await QueryVectorDB(request.question, history);
                        var responseObj = JsonConvert.DeserializeObject<JObject>(vectorResponse);
                        if (responseObj["answer"] is JObject answerObj)
                        {
                            finalAnswer = answerObj["answer"]?.ToString();
                            finalReferences = answerObj["references"]?.ToObject<List<References>>();
                        }
                        else
                        {
                            finalAnswer = responseObj["answer"]?.ToString();
                            finalReferences = responseObj["references"]?.ToObject<List<References>>();
                        }
                    }
                    catch (Exception ex)
                    {
                        _logFile.MyLogFile("ERROR", $"QueryVectorDB failed: {ex.Message}");
                        finalAnswer = "Sorry, there was an error retrieving your answer from the function app.";
                        finalReferences = null;
                    }
                    break;

                case var _ when agent.Trim() == "GenerateProposal":
                    try
                    {
                        if (string.IsNullOrWhiteSpace(request.uploadedFileBlobName))
                        {
                            throw new InvalidOperationException("GenerateProposal called without uploaded file.");
                        }
                        var proposalResult = await GenerateProposalFromDocument(request.uploadedFileBlobName);
                        finalAnswer = proposalResult;
                        finalReferences = null;
                    }
                    catch (Exception ex)
                    {
                        _logFile.MyLogFile("ERROR", $"GenerateProposal failed: {ex.Message}");
                        finalAnswer = "Sorry, there was an error generating the proposal from your uploaded document.";
                        finalReferences = null;
                    }
                    break;

                default:
                    finalAnswer = "I can not find any information that matches your question.";
                    break;
            }

            // Update conversation history
            var historyObj = await UpdateConversationHistory(history, request.question, finalAnswer, azureOpenAI);

            // Update the blob with the new history
            if (containerClient != null && !string.IsNullOrWhiteSpace(blobPath))
            {
                await UpdateBlobJsonAsync(blobPath, containerClient, historyObj);
            }

            return new OpenAIChatResponse
            {
                answer = finalAnswer,
                references = finalReferences
            };
        }

        // Vector DB call
        public async Task<string> QueryVectorDB(string question, List<ConversationHistory> history)
        {
            // Acquire token
            var app = ConfidentialClientApplicationBuilder.Create(clientId)
                .WithClientSecret(clientSecret)
                .WithAuthority(new Uri($"https://login.microsoftonline.com/{tenantId}"))
                .Build();

            var scopes = new[] { $"{functionAppResource}/.default" };
            var authResult = await app.AcquireTokenForClient(scopes).ExecuteAsync();

            var payload = new VectorFunctionPayload
            {
                question = question,
                history = history
            };

            var jsonPayload = JsonConvert.SerializeObject(payload);
            var content = new StringContent(jsonPayload, System.Text.Encoding.UTF8, "application/json");

            // Add Bearer token to request
            _httpClient.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", authResult.AccessToken);

            var response = await _httpClient.PostAsync(
                System.Configuration.ConfigurationManager.AppSettings["VectorQueryEndpoint"],
                content);

            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }

        // Blob fetch - History
        public async Task<string> GetBlobJsonAsync(string blobName, Azure.Storage.Blobs.BlobContainerClient containerClient)
        {
            try
            {
                var blobClient = containerClient.GetBlobClient(blobName);

                if (!await blobClient.ExistsAsync())
                {
                    throw new Exception($"Blob '{blobName}' not found in container '{containerClient.Name}'.");
                }

                var download = await blobClient.DownloadContentAsync();
                return download.Value.Content.ToString(); // JSON string
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error fetching blob: {ex.Message}");
                throw;
            }
        }

        // Upload updated history to blob
        public async Task UpdateBlobJsonAsync(string blobName, Azure.Storage.Blobs.BlobContainerClient containerClient, List<ConversationHistory> updatedHistory)
        {
            var blobClient = containerClient.GetBlobClient(blobName);
            string json = JsonConvert.SerializeObject(updatedHistory);
            using (var stream = new System.IO.MemoryStream(System.Text.Encoding.UTF8.GetBytes(json)))
            {
                await blobClient.UploadAsync(stream, overwrite: true);
            }
        }

        // Helper to update conversation history
        private async Task<List<ConversationHistory>> UpdateConversationHistory(
            List<ConversationHistory> history,
            string question,
            string answer,
            AzureOpenAIAPI azureOpenAI)
        {
            // Get existing summary and recent
            string summary = history != null && history.Count > 0 ? history[0].olderResponseSummary : null;
            List<OpenAIChatMessage> recent = history != null && history.Count > 0 && history[0].recentResponses != null
                ? new List<OpenAIChatMessage>(history[0].recentResponses)
                : new List<OpenAIChatMessage>();

            // Add new Q/A
            recent.Add(new OpenAIChatMessage { role = "user", content = question });
            recent.Add(new OpenAIChatMessage { role = "assistant", content = answer });

            // If recent has grown beyond 2 pairs, roll oldest into summary (batch all at once)
            if (recent.Count > 4)
            {
                var rolledOff = recent.Take(recent.Count - 4).ToList();
                recent = recent.Skip(recent.Count - 4).ToList();

                var summarizePrompt = new List<OpenAIChatMessage>
                {
                    new OpenAIChatMessage
                    {
                        role = "system",
                        content =
                                "Your job is to compress chat history into a short internal memory.\n" +
                                "- Combine the existing summary (if any) with the newly older chats.\n" +
                                "- Provide a brief summary of the previous interactions to maintain context" +
                                "- Summarize the key question and relevant context from the history in 2-3 sentences\n" +
                                "- No headings, no intro text, no concluding sentence.\n" +
                                "- Do NOT say 'Summary', 'Updated Summary', or similar.\n" +
                                "- Focus on stable facts, user preferences, and long-term context.\n" +
                                "- Ignore chit-chat, apologies, generic explanations, or transient issues."


                    }
                };

                if (!string.IsNullOrWhiteSpace(summary))
                {
                    summarizePrompt.Add(new OpenAIChatMessage
                    {
                        role = "user",
                        content = $"Existing summary:\n{summary}"
                    });
                }

                summarizePrompt.Add(new OpenAIChatMessage
                {
                    role = "user",
                    content = $"Newly older chat(s):\n{JsonConvert.SerializeObject(rolledOff)}"
                });

                var summaryResult = await azureOpenAI.CallAzureOpenAI(summarizePrompt);
                summary = summaryResult;
            }

            // Build final history object
            return new List<ConversationHistory>
            {
                new ConversationHistory 
                {
                    olderResponseSummary = summary,
                    recentResponses = recent
                }
            };
        }

        // --- Helper method for proposal generation workflow ---
        private async Task<string> GenerateProposalFromDocument(string blobName)
        {
            // Acquire token
            var app = ConfidentialClientApplicationBuilder.Create(clientId)
                .WithClientSecret(clientSecret)
                .WithAuthority(new Uri($"https://login.microsoftonline.com/{tenantId}"))
                .Build();

            var scopes = new[] { $"{functionAppResource}/.default" };
            var authResult = await app.AcquireTokenForClient(scopes).ExecuteAsync();

            // Download the file from blob storage
            var blobServiceClient = new Azure.Storage.Blobs.BlobServiceClient(blobStorageConnectionString);
            var containerClient = blobServiceClient.GetBlobContainerClient("ai-agent-history");
            var blobClient = containerClient.GetBlobClient(blobName);

            if (!await blobClient.ExistsAsync())
                throw new Exception($"Blob '{blobName}' not found.");

            var download = await blobClient.DownloadContentAsync();
            var fileBytes = download.Value.Content.ToArray();

            // Determine MIME type based on file extension
            string mimeType = "application/octet-stream";
            if (blobName.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase))
                mimeType = "application/pdf";
            else if (blobName.EndsWith(".docx", StringComparison.OrdinalIgnoreCase))
                mimeType = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
            else if (blobName.EndsWith(".doc", StringComparison.OrdinalIgnoreCase))
                mimeType = "application/msword";

            // Prepare multipart/form-data content
            var multipartContent = new MultipartFormDataContent();
            var fileContent = new ByteArrayContent(fileBytes);
            fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(mimeType);
            var fileNameOnly = System.IO.Path.GetFileName(blobName);
            multipartContent.Add(fileContent, "file", fileNameOnly);

            // Add Bearer token to request
            _httpClient.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", authResult.AccessToken);

            var response = await _httpClient.PostAsync(
                System.Configuration.ConfigurationManager.AppSettings["GenerateProposalEndpoint"],
                multipartContent);

            response.EnsureSuccessStatusCode();

            var contentType = response.Content.Headers.ContentType?.MediaType;
            if (string.Equals(contentType, "application/pdf", StringComparison.OrdinalIgnoreCase))
            {
                var pdfBytes = await response.Content.ReadAsByteArrayAsync();

                // Upload PDF bytes to blob storage
                var generatedBlobName = $"generated/{Guid.NewGuid()}.pdf";
                var generatedBlobClient = containerClient.GetBlobClient(generatedBlobName);

                using (var stream = new System.IO.MemoryStream(pdfBytes))
                {
                    await generatedBlobClient.UploadAsync(stream, overwrite: true);
                }

                // Return the blob name or URL
                return generatedBlobName; // or blobClient.Uri.ToString() for full URL
            }

            return await response.Content.ReadAsStringAsync();
        }
    }
}
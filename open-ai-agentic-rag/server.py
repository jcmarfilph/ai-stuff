import json
import time
from http.server import BaseHTTPRequestHandler
from agent_engine import CognitiveAgentEngine

# Instantiate the cognitive orchestrator singleton
agent_runtime = CognitiveAgentEngine()

class OpenRouterAgentGatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # Mutes background network logging

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode('utf-8'))
                messages = payload.get("messages", [])
                requested_model = payload.get("model", "scratch-agent-v1")
                
                user_prompt = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_prompt = msg.get("content", "")
                        break
                
                # Execute the full cognitive agentic lifecycle loop
                agent_outcome = agent_runtime.process_request(user_prompt, requested_model)
                
                # Attempt to parse the content string back into a structural dictionary
                # This safely eliminates escaping by treating it as an object rather than a raw string
                try:
                    content_payload = json.loads(agent_outcome["final_answer"])
                except (json.JSONDecodeError, TypeError):
                    content_payload = agent_outcome["final_answer"]

                # Construct an unescaped, native OpenRouter structure object payload
                openai_compliant_payload = {
                    "id": f"chatcmpl-agentic-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": requested_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content_payload  # Directly maps the unescaped payload object
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(user_prompt.split()),
                        "completion_tokens": len(str(content_payload).split()),
                        "total_tokens": len(user_prompt.split()) + len(str(content_payload).split())
                    }
                }
                
                # Serialize the entire combined API gateway framework response at once
                final_output_bytes = json.dumps(openai_compliant_payload, indent=2, ensure_ascii=False).encode('utf-8')

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(final_output_bytes)
                print(f"📡 [OpenRouter Gateway] 200 OK - Native unescaped JSON object sent for: '{requested_model}'")

            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error_response = {"error": {"message": str(e), "type": "agent_runtime_error"}}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_msg = {"error": {"message": f"Path '{self.path}' not found", "type": "invalid_route_error"}}
            self.wfile.write(json.dumps(err_msg).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

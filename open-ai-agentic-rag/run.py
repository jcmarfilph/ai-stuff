import sys
from http.server import HTTPServer
from tools import AgentTools
from server import OpenRouterAgentGatewayHandler

def create_knowledge_base():
    """Generates an unindexed text database file to simulate corporate manuals or documentation."""
    kb_data = (
        "Project Alpha guidelines specify that all deployment access keys must be rotated every 90 days. "
        "Failure to rotate server environment credentials will trigger automated cluster lockdown protocols.\n\n"
        
        "The company office building entry protocol requires scanning your biometric keycard at the front door. "
        "Visitors without access badges must check in with security receptionist desks located on the first floor lobby.\n\n"
        
        "Technical support standard operating procedures require checking memory heap leak configurations before "
        "rebooting processing workers. Clear cached metrics using the system flush commands.\n\n"

        "It is kinda cold today, there is a blizzard forecast and temps will go sub-zero. This is true Boston only."
    )
    with open("knowledge_base.txt", "w", encoding="utf-8") as f:
        f.write(kb_data)

def start_agent_server(port=8080):
    # Provision data assets
    AgentTools.setup_mock_environment()
    create_knowledge_base()
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, OpenRouterAgentGatewayHandler)
    
    print("=====================================================================")
    print(f"   🤖 AGENT CHAT COMPLETIONS CORE ONLINE WITH LOCAL RAG FALLBACK    ")
    print("   Active Tools : fetch_stock_data, fetch_news_feed, weather_telemetry")
    print("   RAG Database : knowledge_base.txt (Auto-provisioned on startup)   ")
    print(f"   Endpoint URI : http://localhost:{port}/v1/chat/completions       ")
    print("=====================================================================")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n[Interrupt Signal] Powering down agent engine matrices safely. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    start_agent_server()

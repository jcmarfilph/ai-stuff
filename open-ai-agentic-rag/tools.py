import os

class AgentTools:
    """Encapsulated tool registry executing actions on local text files."""
    
    @staticmethod
    def setup_mock_environment():
        """Creates the disk files used as underlying databases."""
        files = {
            "stocks.txt": "AAPL:175.50\nTSLA:180.20\nNVDA:875.12\nMSFT:420.30",
            "news.txt": "BREAKING: Tech sector index hits record high.\nEVENT: Annual DevCon starting tonight.",
            "weather.txt": "CITY: New York\nTEMP: 72F\nCOND: Partly Cloudy\nRAIN: 10%\nCITY: Boston\nTEMP: 42F\nCOND: Partly Cloudy\nRAIN: 0%\nSNOW: 12in"
        }
        for filename, contents in files.items():
            with open(filename, "w", encoding="utf-8") as f:
                f.write(contents)

    @staticmethod
    def fetch_stock_data() -> str:
        return AgentTools._read("stocks.txt")

    @staticmethod
    def fetch_news_feed() -> str:
        return AgentTools._read("news.txt")

    @staticmethod
    def fetch_weather_telemetry() -> str:
        return AgentTools._read("weather.txt")

    @staticmethod
    def _read(filename: str) -> str:
        if not os.path.exists(filename):
            return f"Error: Tool data asset repository '{filename}' is offline."
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()

    @classmethod
    def execute_tool_by_name(cls, name: str) -> str:
        """Dynamic functional execution routing layer."""
        method = getattr(cls, name, None)
        if method and callable(method) and name != "execute_tool_by_name":
            return method()
        return "Error: Requested tool function signature not present in registry."

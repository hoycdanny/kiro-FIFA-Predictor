"""
Tests for the MCP server entry point (src/server.py).

Verifies:
- Server module can be imported without errors
- create_server() returns a FastMCP instance
- All 6 MCP tools are registered
- startup() initializes all dependencies correctly
"""

import pytest
from unittest.mock import patch, MagicMock

from src.server import mcp, create_server


class TestServerImport:
    """Test that the server module imports cleanly."""

    def test_server_module_imports(self):
        """Server module should import without errors."""
        import src.server
        assert src.server is not None

    def test_mcp_instance_exists(self):
        """Module-level mcp instance should exist."""
        assert mcp is not None

    def test_create_server_returns_fastmcp(self):
        """create_server() should return the FastMCP server instance."""
        server = create_server()
        assert server is mcp

    def test_mcp_server_name(self):
        """Server should be named 'fifa-predictor'."""
        assert mcp.name == "fifa-predictor"


class TestToolRegistration:
    """Test that all 6 MCP tools are registered."""

    def test_predict_match_tool_exists(self):
        """predict_match tool should be registered."""
        tools = mcp._tool_manager._tools
        assert "predict_match" in tools

    def test_predict_group_tool_exists(self):
        """predict_group tool should be registered."""
        tools = mcp._tool_manager._tools
        assert "predict_group" in tools

    def test_predict_champion_tool_exists(self):
        """predict_champion tool should be registered."""
        tools = mcp._tool_manager._tools
        assert "predict_champion" in tools

    def test_update_results_tool_exists(self):
        """update_results tool should be registered."""
        tools = mcp._tool_manager._tools
        assert "update_results" in tools

    def test_accuracy_stats_tool_exists(self):
        """accuracy_stats tool should be registered."""
        tools = mcp._tool_manager._tools
        assert "accuracy_stats" in tools

    def test_team_info_tool_exists(self):
        """team_info tool should be registered."""
        tools = mcp._tool_manager._tools
        assert "team_info" in tools

    def test_exactly_six_tools_registered(self):
        """Exactly 6 tools should be registered."""
        tools = mcp._tool_manager._tools
        assert len(tools) == 6


class TestStartup:
    """Test the startup initialization flow."""

    @pytest.mark.asyncio
    async def test_startup_initializes_globals(self):
        """startup() should initialize all module-level globals."""
        from src.server import startup, _data_manager

        # Before startup, globals might be None from a fresh import
        # Run startup to verify it works with valid data
        await startup()

        import src.server
        assert src.server._data_manager is not None
        assert src.server._prediction_engine is not None
        assert src.server._monte_carlo is not None
        assert src.server._output_formatter is not None
        assert src.server._accuracy_tracker is not None
        assert src.server._input_validator is not None

    @pytest.mark.asyncio
    async def test_startup_aborts_on_invalid_data(self):
        """startup() should abort with SystemExit on data validation failure."""
        import src.server
        from src.data.data_manager import DataValidationError

        with patch.object(
            src.server, "_get_data_dir", return_value=MagicMock()
        ):
            with patch(
                "src.server.DataManager"
            ) as mock_dm_class:
                mock_dm = MagicMock()
                mock_dm.validate_startup.side_effect = DataValidationError(
                    "Test: Missing teams"
                )
                mock_dm_class.return_value = mock_dm

                with pytest.raises(SystemExit) as exc_info:
                    await src.server.startup()

                assert exc_info.value.code == 1

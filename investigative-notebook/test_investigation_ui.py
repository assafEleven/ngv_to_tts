import pytest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets
import types

from NEW_Full_UI_Cell_8 import InvestigationUI

def make_openai_response(json_str):
    mock_choice = MagicMock()
    mock_choice.message.content = json_str
    return MagicMock(choices=[mock_choice])

def make_bq_df(rows=1, cols=1):
    df = MagicMock()
    df.shape = (rows, cols)
    return df

@pytest.fixture
def ui():
    ui = InvestigationUI()
    # Patch display to avoid actual notebook output
    ui.display_conversation = lambda: None
    return ui

# --- NL Query Types ---
def test_simple_retrieval(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai, \
         patch('NEW_Full_UI_Cell_8.bigquery.Client') as mock_bq:
        mock_openai.return_value = make_openai_response('{"sql": "SELECT * FROM table", "explanation": "All rows."}')
        mock_bq.return_value.query.return_value.result.return_value.to_dataframe.return_value = make_bq_df(5, 3)
        ui.nl_query.value = "show all rows from tts_usage_partitioned"
        ui.on_run_nl()
        assert "Complete" in ui.status_label.value


def test_filtered_query_text(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai, \
         patch('NEW_Full_UI_Cell_8.bigquery.Client') as mock_bq:
        mock_openai.return_value = make_openai_response('{"sql": "SELECT * FROM table WHERE text LIKE \'%scam%\'", "explanation": "Filtered by text."}')
        mock_bq.return_value.query.return_value.result.return_value.to_dataframe.return_value = make_bq_df(2, 3)
        ui.nl_query.value = "find all tts generations containing the word scam"
        ui.on_run_nl()
        assert "Complete" in ui.status_label.value


def test_filtered_query_user(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai, \
         patch('NEW_Full_UI_Cell_8.bigquery.Client') as mock_bq:
        mock_openai.return_value = make_openai_response('{"sql": "SELECT * FROM table WHERE user_uid = \'abc\'", "explanation": "Filtered by user."}')
        mock_bq.return_value.query.return_value.result.return_value.to_dataframe.return_value = make_bq_df(1, 3)
        ui.nl_query.value = "show all generations by user abc"
        ui.on_run_nl()
        assert "Complete" in ui.status_label.value


def test_filtered_query_date(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai, \
         patch('NEW_Full_UI_Cell_8.bigquery.Client') as mock_bq:
        mock_openai.return_value = make_openai_response('{"sql": "SELECT * FROM table WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)", "explanation": "Filtered by date."}')
        mock_bq.return_value.query.return_value.result.return_value.to_dataframe.return_value = make_bq_df(3, 3)
        ui.nl_query.value = "tts generations from the past week"
        ui.on_run_nl()
        assert "Complete" in ui.status_label.value


def test_aggregation_count(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai, \
         patch('NEW_Full_UI_Cell_8.bigquery.Client') as mock_bq:
        mock_openai.return_value = make_openai_response('{"sql": "SELECT COUNT(*) FROM table", "explanation": "Count rows."}')
        mock_bq.return_value.query.return_value.result.return_value.to_dataframe.return_value = make_bq_df(1, 1)
        ui.nl_query.value = "how many tts generations in total?"
        ui.on_run_nl()
        assert "Complete" in ui.status_label.value


def test_aggregation_groupby(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai, \
         patch('NEW_Full_UI_Cell_8.bigquery.Client') as mock_bq:
        mock_openai.return_value = make_openai_response('{"sql": "SELECT user_uid, COUNT(*) FROM table GROUP BY user_uid", "explanation": "Count per user."}')
        mock_bq.return_value.query.return_value.result.return_value.to_dataframe.return_value = make_bq_df(10, 2)
        ui.nl_query.value = "show tts generations count per user"
        ui.on_run_nl()
        assert "Complete" in ui.status_label.value


def test_multi_table_join(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai, \
         patch('NEW_Full_UI_Cell_8.bigquery.Client') as mock_bq:
        mock_openai.return_value = make_openai_response('{"sql": "SELECT a.*, b.* FROM table1 a JOIN table2 b ON a.user_uid = b.user_uid", "explanation": "Join tables."}')
        mock_bq.return_value.query.return_value.result.return_value.to_dataframe.return_value = make_bq_df(5, 6)
        ui.nl_query.value = "join tts generations with user info"
        ui.on_run_nl()
        assert "Complete" in ui.status_label.value

# --- Edge Cases ---
def test_ambiguous_query(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai:
        mock_openai.return_value = make_openai_response('{"explanation": "Query is ambiguous.", "follow_up_question": "Which user or time period?", "suggested_query": "Show all tts generations from the last 7 days."}')
        ui.nl_query.value = "show me stuff"
        ui.on_run_nl()
        assert "ambiguous" in str(ui.nl_output.outputs[0]).lower() or "follow_up" in str(ui.nl_output.outputs[0]).lower()


def test_meta_question_df_name(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai:
        mock_openai.return_value = make_openai_response('{"explanation": "The DataFrame is named df in the Python code."}')
        ui.nl_query.value = "What is the name of the DataFrame for this result?"
        ui.on_run_nl()
        assert "df in the Python code" in str(ui.nl_output.outputs[0])


def test_empty_query(ui):
    ui.nl_query.value = ""
    ui.on_run_nl()
    assert "Please enter" in ui.status_label.value


def test_malformed_query(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create', side_effect=Exception("Malformed")):
        ui.nl_query.value = "malformed query"
        ui.on_run_nl()
        assert "Error" in ui.status_label.value

# --- DataFrame Naming ---
def test_custom_df_name(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai, \
         patch('NEW_Full_UI_Cell_8.bigquery.Client') as mock_bq:
        mock_openai.return_value = make_openai_response('{"sql": "SELECT 1", "explanation": "Test"}')
        mock_bq.return_value.query.return_value.result.return_value.to_dataframe.return_value = make_bq_df(1, 1)
        ui.nl_query.value = "find all tts generations - save the df as TestDF"
        ui.on_run_nl()
        assert "TestDF" in globals()

# --- Error Handling ---
def test_openai_timeout(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create', side_effect=TimeoutError):
        ui.nl_query.value = "find all tts generations"
        ui.on_run_nl()
        assert "timed out" in ui.status_label.value


def test_openai_non_json(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai:
        mock_openai.return_value = make_openai_response('Just some text, not JSON.')
        ui.nl_query.value = "find all tts generations"
        ui.on_run_nl()
        assert "explanation" in str(ui.nl_output.outputs[0]).lower() or "Just some text" in str(ui.nl_output.outputs[0])


def test_bigquery_error(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai, \
         patch('NEW_Full_UI_Cell_8.bigquery.Client') as mock_bq:
        mock_openai.return_value = make_openai_response('{"sql": "SELECT * FROM table", "explanation": "All rows."}')
        mock_bq.return_value.query.side_effect = Exception("BigQuery error")
        ui.nl_query.value = "show all rows from tts_usage_partitioned"
        ui.on_run_nl()
        assert "Error" in ui.status_label.value

# --- Conversation Memory ---
def test_followup_query(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai, \
         patch('NEW_Full_UI_Cell_8.bigquery.Client') as mock_bq:
        # First query
        mock_openai.side_effect = [
            make_openai_response('{"sql": "SELECT * FROM table WHERE text LIKE \'%ted cruz%\'", "explanation": "Filtered by text."}'),
            make_openai_response('{"explanation": "The DataFrame is named df in the Python code."}')
        ]
        mock_bq.return_value.query.return_value.result.return_value.to_dataframe.return_value = make_bq_df(2, 3)
        ui.nl_query.value = "find all tts generations with the name ted cruz"
        ui.on_run_nl()
        ui.nl_query.value = "What is the name of the DataFrame for this result?"
        ui.on_run_nl()
        assert "df in the Python code" in str(ui.nl_output.outputs[0])

# --- Stop Button ---
def test_stop_button(ui):
    ui.stop_requested = True
    ui.nl_query.value = "find all tts generations"
    ui.on_run_nl()
    assert "Stopped" in ui.status_label.value

# --- UI/Output ---
def test_long_explanation_scrollable(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai:
        long_text = "A" * 1000
        mock_openai.return_value = make_openai_response(f'{{"explanation": "{long_text}"}}')
        ui.nl_query.value = "meta question"
        ui.on_run_nl()
        # Should display a scrollable widget
        assert any(isinstance(o, widgets.Textarea) for o in ui.nl_output.outputs)

# --- Suggestions ---
def test_suggested_query(ui):
    with patch('NEW_Full_UI_Cell_8.client.chat.completions.create') as mock_openai:
        mock_openai.return_value = make_openai_response('{"explanation": "Ambiguous.", "suggested_query": "Show all tts generations from the last 7 days."}')
        ui.nl_query.value = "show me stuff"
        ui.on_run_nl()
        assert "suggested_query" in str(ui.nl_output.outputs[0]) or "suggested" in str(ui.nl_output.outputs[0]).lower() 
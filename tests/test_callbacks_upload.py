import unittest
from types import SimpleNamespace
from unittest import mock

from callbacks.upload import confirm_delete_session, toggle_delete_session_modal


class UploadCallbackTests(unittest.TestCase):
    def test_toggle_delete_session_modal_opens_for_clicked_session(self):
        fake_ctx = SimpleNamespace(
            triggered=[{"prop_id": '{"index":"session-1","type":"delete-session-btn"}.n_clicks'}],
            triggered_id={"type": "delete-session-btn", "index": "session-1"},
        )
        session = SimpleNamespace(session_id="session-1", session_name="Demo Session")

        with mock.patch("callbacks.upload.callback_context", fake_ctx), \
             mock.patch("callbacks.upload.get_session", return_value=session):
            is_open, body, target, click_store = toggle_delete_session_modal(
                [1, 0],
                None,
                [
                    {"type": "delete-session-btn", "index": "session-1"},
                    {"type": "delete-session-btn", "index": "session-2"},
                ],
                {},
            )

        self.assertTrue(is_open)
        self.assertEqual(target, {"session_id": "session-1"})
        self.assertEqual(click_store, {"session-1": 1})
        self.assertEqual(body.children[0].children, 'Delete session "Demo Session"?')

    def test_toggle_delete_session_modal_closes_on_cancel(self):
        fake_ctx = SimpleNamespace(
            triggered=[{"prop_id": "delete-session-cancel.n_clicks"}],
            triggered_id="delete-session-cancel",
        )

        with mock.patch("callbacks.upload.callback_context", fake_ctx):
            is_open, body, target, click_store = toggle_delete_session_modal([], 1, [], {"session-1": 1})

        self.assertFalse(is_open)
        self.assertEqual(body, "")
        self.assertEqual(target, {})
        self.assertEqual(click_store, mock.ANY)

    def test_confirm_delete_session_refreshes_sessions(self):
        with mock.patch("callbacks.upload.delete_session") as delete_session:
            is_open, refresh = confirm_delete_session(1, {"session_id": "session-1"}, 2)

        delete_session.assert_called_once_with("session-1")
        self.assertFalse(is_open)
        self.assertEqual(refresh, 3)
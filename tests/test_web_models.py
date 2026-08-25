import os
import unittest
from unittest.mock import AsyncMock, patch

import demo_app_with_front
from web_models import SearchRequest


class SearchRequestSecurityTests(unittest.TestCase):
    def test_google_serper_key_is_redacted_from_request_representation(self):
        secret = "SPAR_TEST_SERPER_SECRET"

        request = SearchRequest(
            queries=["retrieval augmented generation"],
            google_serper_key=secret,
        )

        self.assertNotIn(secret, repr(request))
        self.assertNotIn(secret, str(request))
        self.assertEqual(
            request.google_serper_key.get_secret_value(),
            secret,
        )


class SearchEndpointLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_request_logs_do_not_contain_google_serper_key(self):
        secret = "SPAR_TEST_SERPER_SECRET"
        request = SearchRequest(
            queries=["retrieval augmented generation"],
            google_serper_key=secret,
            use_advanced_search=False,
        )

        with patch.object(demo_app_with_front, "logger") as logger:
            with patch.object(
                demo_app_with_front,
                "_simple_search",
                new=AsyncMock(return_value={"status": "ok"}),
            ):
                with patch.dict(os.environ, {}, clear=False):
                    await demo_app_with_front.search_papers(request)

        rendered_messages = []
        for call in logger.info.call_args_list:
            template, *values = call.args
            rendered_messages.append(template % tuple(values) if values else template)

        self.assertNotIn(secret, "\n".join(rendered_messages))

    async def test_request_scoped_google_serper_key_is_removed_after_search(self):
        request = SearchRequest(
            queries=["retrieval augmented generation"],
            google_serper_key="SPAR_TEST_SERPER_SECRET",
            use_advanced_search=False,
        )

        with patch.object(
            demo_app_with_front,
            "_simple_search",
            new=AsyncMock(return_value={"status": "ok"}),
        ):
            with patch.dict(os.environ, {}, clear=True):
                await demo_app_with_front.search_papers(request)
                self.assertNotIn("GOOGLE_SERPER_KEY", os.environ)

    async def test_server_google_serper_key_is_restored_after_search(self):
        request = SearchRequest(
            queries=["retrieval augmented generation"],
            google_serper_key="SPAR_TEST_REQUEST_SECRET",
            use_advanced_search=False,
        )

        with patch.object(
            demo_app_with_front,
            "_simple_search",
            new=AsyncMock(return_value={"status": "ok"}),
        ):
            with patch.dict(
                os.environ,
                {"GOOGLE_SERPER_KEY": "SPAR_TEST_SERVER_SECRET"},
                clear=True,
            ):
                await demo_app_with_front.search_papers(request)
                self.assertEqual(
                    os.environ["GOOGLE_SERPER_KEY"],
                    "SPAR_TEST_SERVER_SECRET",
                )


if __name__ == "__main__":
    unittest.main()

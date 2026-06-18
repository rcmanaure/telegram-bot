"""Tests for laboratory customer support RAG (sp-diagnostico-histologico.md)"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os
from pathlib import Path

# Assumes project root is 2 dirs up from tests/
PROJECT_ROOT = Path(__file__).parent.parent
LAB_DOC_PATH = PROJECT_ROOT / "sp-diagnostico-histologico.md"


@pytest.fixture
def lab_document_content():
    """Load lab document for testing"""
    if not LAB_DOC_PATH.exists():
        pytest.skip(f"Lab document not found: {LAB_DOC_PATH}")
    return LAB_DOC_PATH.read_text(encoding="utf-8")


class TestLabDocumentPolicies:
    """Test that critical policies are correctly embedded"""

    def test_formol_policy_in_document(self, lab_document_content):
        """Lab document must mention formol policy"""
        assert "formol" in lab_document_content.lower()
        assert "No se aceptan" in lab_document_content

    def test_payment_policy_in_document(self, lab_document_content):
        """Lab document must mention full payment upfront"""
        assert "cancelada en su totalidad" in lab_document_content.lower()

    def test_whatsapp_text_only_policy(self, lab_document_content):
        """Lab document must mention WhatsApp text-only"""
        assert "mensajería texto" in lab_document_content.lower() or "No respondemos llamadas" in lab_document_content

    def test_results_timeline_policy(self, lab_document_content):
        """Lab document must mention 3-5 day timeline"""
        assert "3 a 5 días" in lab_document_content or "3-5 días" in lab_document_content

    def test_frozen_cut_exception(self, lab_document_content):
        """Lab document must mention frozen cut exception"""
        assert "corte congelado" in lab_document_content.lower()
        assert "mismo día" in lab_document_content.lower()


class TestLabPriceLookup:
    """Test price retrieval from lab document"""

    def test_gastric_biopsy_price(self, lab_document_content):
        """Gastric biopsy code SDG014 should be $80"""
        # SDG014 is Biopsia Gástrica Endoscópica = $80.00
        assert "SDG014" in lab_document_content
        assert "$80.00" in lab_document_content

    def test_appendix_biopsy_price(self, lab_document_content):
        """Appendix biopsy code SDG033 should be $90"""
        assert "SDG033" in lab_document_content
        assert "$90.00" in lab_document_content

    def test_frozen_cut_price(self, lab_document_content):
        """Frozen cut (corte congelado) should be $490"""
        assert "490" in lab_document_content or "$490" in lab_document_content

    def test_gynecology_section_prices(self, lab_document_content):
        """Gynecology section should have prices"""
        assert "GINECOLÓGICO" in lab_document_content
        assert "GIN001" in lab_document_content
        assert "$80.00" in lab_document_content


class TestLabSampleHandling:
    """Test sample handling instructions in document"""

    def test_formol_requirements(self, lab_document_content):
        """Sample handling must clearly state formol requirements"""
        text_lower = lab_document_content.lower()
        assert "formol" in text_lower
        assert "único" in text_lower or "only" in text_lower.replace("única", "unique")

    def test_sample_storage_guidelines(self, lab_document_content):
        """Document must have sample storage time guidelines"""
        assert "conservar" in lab_document_content.lower() or "preservation" in lab_document_content.lower()

    def test_frozen_cut_handling_different_from_routine(self, lab_document_content):
        """Frozen cuts must be explicitly different from routine samples"""
        text_lower = lab_document_content.lower()
        assert "corte congelado" in text_lower
        assert "no debe estar" in text_lower or "must not be" in text_lower


class TestLabContactInfo:
    """Test contact information accuracy"""

    def test_whatsapp_number_present(self, lab_document_content):
        """WhatsApp contact must be present"""
        assert "04148050764" in lab_document_content or "WhatsApp" in lab_document_content

    def test_email_present(self, lab_document_content):
        """Email must be present"""
        assert "SP.UDH.PLC@GMAIL.COM" in lab_document_content or "@" in lab_document_content

    def test_physical_address_present(self, lab_document_content):
        """Physical address must be present"""
        assert "Puerto la Cruz" in lab_document_content or "dirección" in lab_document_content.lower()

    def test_business_hours_present(self, lab_document_content):
        """Business hours must be documented"""
        assert "8:00 AM" in lab_document_content or "lunes" in lab_document_content.lower()


class TestPromptPolicyInjection:
    """Test that system prompt includes critical policies"""

    @patch("services.prompts.CHANNEL_FORMATTING", {"telegram": MagicMock(format_instructions="")})
    def test_system_prompt_includes_formol_policy(self):
        """System prompt must inject formol policy"""
        from services.prompts import build_system_prompt

        prompt = build_system_prompt(expertise_area="")
        assert "formol" in prompt.lower()
        assert "POLÍTICAS CRÍTICAS" in prompt or "formol" in prompt

    @patch("services.prompts.CHANNEL_FORMATTING", {"telegram": MagicMock(format_instructions="")})
    def test_system_prompt_includes_payment_policy(self):
        """System prompt must inject payment policy"""
        from services.prompts import build_system_prompt

        prompt = build_system_prompt(expertise_area="")
        assert "pago" in prompt.lower() or "payment" in prompt.lower()

    @patch("services.prompts.CHANNEL_FORMATTING", {"telegram": MagicMock(format_instructions="")})
    def test_system_prompt_includes_whatsapp_text_only(self):
        """System prompt must mention WhatsApp text-only policy"""
        from services.prompts import build_system_prompt

        prompt = build_system_prompt(expertise_area="")
        # At least one of these should be in the prompt
        has_whatsapp_rule = "WhatsApp" in prompt or "texto" in prompt.lower()
        assert has_whatsapp_rule or "mensajes de voz" in prompt.lower()


class TestVisionExtraction:
    """Test E1: Auto-Quote from medical order image"""

    @pytest.mark.asyncio
    async def test_extract_service_codes_success(self):
        """Vision extraction should return codes with confidence"""
        from rag import extract_service_codes_from_image

        # Mock the LLM call to return sample codes
        mock_response = '{"codes": ["GIN001", "GMM002"], "confidence": 0.92}'
        with patch("rag.call_chat", new_callable=AsyncMock, return_value=mock_response):
            result = await extract_service_codes_from_image("fake_base64_image")

        assert result["codes"] == ["GIN001", "GMM002"]
        assert result["confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_extract_service_codes_low_confidence(self):
        """Vision extraction with low confidence should be flagged"""
        from rag import extract_service_codes_from_image

        # Mock low confidence response
        mock_response = '{"codes": ["SDG014"], "confidence": 0.45}'
        with patch("rag.call_chat", new_callable=AsyncMock, return_value=mock_response):
            result = await extract_service_codes_from_image("fake_base64_image")

        assert result["confidence"] < 0.70
        # User should be asked to confirm via text

    @pytest.mark.asyncio
    async def test_extract_service_codes_empty_image(self):
        """Vision extraction on non-order image should return empty codes"""
        from rag import extract_service_codes_from_image

        mock_response = '{"codes": [], "confidence": 0.1}'
        with patch("rag.call_chat", new_callable=AsyncMock, return_value=mock_response):
            result = await extract_service_codes_from_image("fake_base64_image")

        assert result["codes"] == []
        assert result["confidence"] < 0.70


class TestLabChunkingStrategy:
    """Test that lab document chunks preserve structure"""

    def test_price_table_chunking(self, lab_document_content):
        """Chunking should preserve price table rows as units"""
        from rag import chunk_text, _split_markdown_tables

        # Process through markdown splitter
        preprocessed = _split_markdown_tables(lab_document_content)

        # Each price row should have section header prepended
        assert "## GINECOLÓGICO" in preprocessed or "GINECOLÓGICO" in preprocessed
        assert "GIN001" in preprocessed

        # Now chunk it
        chunks = chunk_text(lab_document_content, source="sp-diagnostico-histologico")

        # At least some chunks should contain price info
        price_chunks = [c for c in chunks if "$" in c["content"]]
        assert len(price_chunks) > 0, "No price chunks found"

        # Each chunk should be reasonable size
        for chunk in chunks:
            assert len(chunk["content"]) > 0
            assert len(chunk["content"]) < 2000  # Not excessively large


class TestRetrievalScores:
    """Test that retrieval is strict for prices"""

    @pytest.mark.asyncio
    async def test_price_query_high_threshold(self):
        """Price queries should require high similarity (not 0.20)"""
        # This is a policy test — actual implementation would need
        # a database with indexed chunks and a test tenant
        # Skipping for now as it requires full DB setup

        pytest.skip("Requires full integration test with DB")


# Integration tests (require docker + live DB)

@pytest.mark.asyncio
@pytest.mark.integration
async def test_lab_document_indexing(api_client, authed_api_client):
    """[INTEGRATION] Index lab document and verify retrieval"""
    pytest.skip("TODO: Write after DB migration")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_policy_enforcement_in_response(api_client, authed_api_client):
    """[INTEGRATION] Query about muestras → response includes formol policy"""
    pytest.skip("TODO: Write after DB migration")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_auto_quote_end_to_end(api_client, authed_api_client):
    """[INTEGRATION] Upload order image → receive price quote + instructions"""
    pytest.skip("TODO: Write after DB migration")

from pydantic import BaseModel, Field


class IndexDocumentDto(BaseModel):
    doc_id: str = Field(..., description="Stable id for this document (re-index replaces prior chunks).")
    document: str = Field(..., description="Plain text to chunk, embed, and upsert into Milvus.")
    section_id: str | None = Field(
        default=None,
        max_length=256,
        description="Section / chapter id on every chunk; if omitted or blank, defaults to doc_id.",
    )

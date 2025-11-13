# AI Document Intelligence Platform — API v1 Specification

**Version:** 1.0  
**Status:** Draft  
**Base URL (local):** `http://localhost:8000/api/v1`  
**Media Type:** `application/json`  
**Authentication:** None (Phase 1 — will add JWT in v2)  

This document defines the **API contract** for the AI Document Intelligence Platform.  
It specifies resources, endpoints, parameters, and response formats for version **1** of the API.

The purpose is to ensure a consistent, predictable, and stable interface between backend,
frontend, AI pipelines, and external integrations.

---

# 1. Conventions

## 1.1 JSON Field Naming
All JSON uses **snake_case**.

## 1.2 Monetary Values
All amounts and rates are returned as **strings**, not floats, to avoid rounding errors.

Example:
```json
"amount_gross": "1586.72"
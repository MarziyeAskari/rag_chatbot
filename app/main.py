"""
FastAPI application for RAG Chatbot
"""

from fastapi import FastAPI
from src.config_loader import get_setting

app = FastAPI()

setting =get_setting()
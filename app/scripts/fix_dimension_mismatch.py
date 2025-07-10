# scripts/fix_dimension_mismatch.py
"""
Script to fix ChromaDB embedding dimension mismatch issues.
Run this script to reset your ChromaDB collection when you encounter dimension errors.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_embedding_model():
    """Check what embedding dimension your current model produces"""
    try:
        ollama = LLMService()
        test_text = "This is a test sentence to check embedding dimensions."
        
        logger.info("Testing embedding model...")
        embedding = await ollama.create_embedding(test_text)
        
        if embedding:
            logger.info(f"✅ Embedding model working")
            logger.info(f"✅ Current embedding dimension: {len(embedding)}")
            logger.info(f"✅ Model: {settings.EMBEDDING_MODEL}")
            return len(embedding)
        else:
            logger.error("❌ Failed to get embedding from model")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error testing embedding model: {str(e)}")
        return None

async def reset_chroma_collection():
    """Reset the ChromaDB collection to fix dimension issues"""
    try:
        logger.info("Initializing RAG service...")
        rag_service = RAGService()
        
        logger.info("Resetting ChromaDB collection...")
        await rag_service.reset_collection()
        
        logger.info("✅ Collection reset successfully!")
        
        # Clean up
        await rag_service.close()
        
    except Exception as e:
        logger.error(f"❌ Error resetting collection: {str(e)}")
        raise

async def verify_setup():
    """Verify that everything is working correctly"""
    try:
        logger.info("Verifying setup...")
        
        # Check embedding model
        embedding_dim = await check_embedding_model()
        if not embedding_dim:
            return False
            
        # Test RAG service initialization
        rag_service = RAGService()
        await rag_service.initialize()
        
        logger.info("✅ Setup verification completed successfully!")
        
        # Clean up
        await rag_service.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup verification failed: {str(e)}")
        return False

async def main():
    """Main function to fix dimension mismatch issues"""
    print("🔧 ChromaDB Dimension Fix Tool")
    print("=" * 40)
    
    # Step 1: Check current embedding model
    print("\n📊 Step 1: Checking embedding model...")
    embedding_dim = await check_embedding_model()
    
    if not embedding_dim:
        print("❌ Cannot proceed without working embedding model")
        print("Please check your Ollama installation and model configuration")
        return
    
    # Step 2: Show current configuration
    print(f"\n⚙️  Current Configuration:")
    print(f"   - Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"   - Embedding Dimension: {embedding_dim}")
    print(f"   - ChromaDB Path: {settings.CHROMA_PATH}")
    print(f"   - Collection Name: {settings.CHROMA_COLLECTION_NAME}")
    
    # Step 3: Ask user if they want to reset
    print(f"\n🔄 The embedding dimension mismatch error occurs when:")
    print(f"   - Your collection expects one dimension (e.g., 384)")
    print(f"   - Your embedding model produces another (e.g., 1536)")
    print(f"   - Current model produces: {embedding_dim} dimensions")
    
    response = input(f"\n❓ Do you want to reset the ChromaDB collection? (y/N): ").lower()
    
    if response in ['y', 'yes']:
        print(f"\n🗑️  Step 3: Resetting ChromaDB collection...")
        await reset_chroma_collection()
        
        print(f"\n✅ Step 4: Verifying setup...")
        success = await verify_setup()
        
        if success:
            print(f"\n🎉 All done! Your ChromaDB is now ready to use.")
            print(f"   - Collection reset successfully")
            print(f"   - Embedding dimension: {embedding_dim}")
            print(f"   - You can now upload documents without dimension errors")
        else:
            print(f"\n❌ Verification failed. Please check the logs for errors.")
    else:
        print(f"\n📝 To fix this manually:")
        print(f"   1. Delete the ChromaDB directory: {settings.CHROMA_PATH}")
        print(f"   2. Or use a different embedding model with 384 dimensions")
        print(f"   3. Or modify your code to handle {embedding_dim} dimensions")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n👋 Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
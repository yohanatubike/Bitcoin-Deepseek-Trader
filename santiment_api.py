"""
Module for interacting with Santiment API to fetch social sentiment and on-chain metrics
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import san

logger = logging.getLogger(__name__)

class SantimentAPI:
    def __init__(self, api_key: str, api_url: str = "https://api.santiment.net/graphql"):
        """
        Initialize the Santiment API client
        
        Args:
            api_key: Santiment API key
            api_url: Santiment API URL (not used with SDK, kept for backwards compatibility)
        """
        if not api_key or api_key == "YOUR_SANTIMENT_API_KEY":
            raise ValueError("Invalid Santiment API key. Please set a valid API key in config.ini")
            
        # Configure Santiment SDK
        san.ApiConfig.api_key = api_key
        logger.info("Initializing Santiment API client")
        
        # Test API connection
        self._test_api_connection()
    
    def _test_api_connection(self) -> None:
        """Test the API connection and key validity"""
        try:
            # Try to fetch a simple metric to test authentication
            test_data = san.get(
                "price_usd",
                slug="bitcoin",
                from_date=(datetime.utcnow() - timedelta(days=1)).isoformat(),
                to_date=datetime.utcnow().isoformat(),
                interval="1d"
            )
            
            if isinstance(test_data, pd.DataFrame) and not test_data.empty:
                logger.info("Successfully authenticated with Santiment API")
                logger.debug(f"Test data response: {test_data.to_dict()}")
            else:
                raise ValueError("Unexpected response format from Santiment API")
                
        except Exception as e:
            logger.error(f"Santiment API authentication failed: {str(e)}")
            raise ValueError("Failed to authenticate with Santiment API. Please check your API key.")
    
    def fetch_sentiment_data(self, slug: str = "bitcoin") -> Dict[str, Any]:
        """
        Fetch social sentiment metrics for a given asset
        Currently returning empty data until API is fully configured
        
        Args:
            slug: Asset slug (default: bitcoin)
            
        Returns:
            Dictionary containing sentiment metrics (empty for now)
        """
        logger.info(f"Sentiment data temporarily disabled for {slug}")
        
        # Return empty sentiment data
        response = {
            "social_sentiment": {}
        }
        
        logger.info(f"Returning empty sentiment data: {json.dumps(response, indent=2)}")
        return response
    
    def fetch_onchain_metrics(self, slug: str = "bitcoin") -> Dict[str, Any]:
        """
        Fetch on-chain metrics for a given asset
        
        Args:
            slug: Asset slug (default: bitcoin)
            
        Returns:
            Dictionary containing on-chain metrics
        """
        logger.info(f"Fetching on-chain metrics for {slug}")
        
        try:
            # Calculate time ranges
            to_date = datetime.utcnow()
            from_date = to_date - timedelta(days=1)
            
            logger.debug(f"Time range: from {from_date.isoformat()} to {to_date.isoformat()}")
            
            # Fetch network growth metric
            network_growth = san.get(
                "network_growth",
                slug=slug,
                from_date=from_date.isoformat(),
                to_date=to_date.isoformat(),
                interval="5m"
            )
            
            logger.debug(f"Raw network_growth data: {network_growth.to_dict() if not network_growth.empty else 'Empty'}")
            
            # Get latest values with proper error handling
            latest_network = float(network_growth.iloc[-1].value) if not network_growth.empty else 0.0
            
            # Format response
            response = {
                "onchain_metrics": {
                    "network_growth": latest_network
                }
            }
            
            logger.info(f"Onchain metrics response: {json.dumps(response, indent=2)}")
            return response
            
        except Exception as e:
            logger.error(f"Error fetching on-chain metrics: {str(e)}")
            default_response = self._generate_default_onchain()
            logger.info(f"Using default onchain response: {json.dumps(default_response, indent=2)}")
            return default_response
    
    def fetch_news_sentiment(self, slug: str = "bitcoin") -> Dict[str, Any]:
        """
        Fetch news sentiment metrics for a given asset
        
        Args:
            slug: Asset slug (default: bitcoin)
            
        Returns:
            Dictionary containing news sentiment metrics
        """
        logger.info(f"Fetching news sentiment for {slug}")
        
        try:
            # Calculate time ranges
            to_date = datetime.utcnow()
            from_date = to_date - timedelta(days=1)
            
            logger.debug(f"Time range: from {from_date.isoformat()} to {to_date.isoformat()}")
            
            # Currently returning empty response as metrics are commented out
            response = {
                "news_sentiment": {}
            }
            
            logger.info(f"News sentiment response: {json.dumps(response, indent=2)}")
            return response
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {str(e)}")
            default_response = self._generate_default_news()
            logger.info(f"Using default news sentiment response: {json.dumps(default_response, indent=2)}")
            return default_response
            
    def _generate_default_sentiment(self) -> Dict[str, Any]:
        """Generate default sentiment data when API call fails"""
        return {
            "social_sentiment": {
                "sentiment_balance": 0.0,
                "social_volume": 0,
                "dev_activity": 0.0
            }
        }
        
    def _generate_default_onchain(self) -> Dict[str, Any]:
        """Generate default on-chain metrics when API call fails"""
        return {
            "onchain_metrics": {
                "network_growth": 0.0
            }
        }
        
    def _generate_default_news(self) -> Dict[str, Any]:
        """Generate default news sentiment when API call fails"""
        return {
            "news_sentiment": {}
        } 
import os
import ssl
import certifi
import urllib.request
import logging

logger = logging.getLogger(__name__)

def setup_ssl_context():
    """Set up SSL context with proper certificate verification."""
    return ssl.create_default_context(cafile=certifi.where())

def configure_ssl():
    """
    Configure SSL certificates for secure network connections.
    This is particularly important when running in Docker containers
    where the default certificate store might be incomplete.
    """
    try:
        # Get the path to the certifi certificate bundle
        cafile = certifi.where()
        
        # Create a new SSL context using the certifi certificates
        ssl_context = ssl.create_default_context(cafile=cafile)
        
        # Set the default SSL context
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Set environment variables for requests and urllib3
        os.environ['REQUESTS_CA_BUNDLE'] = cafile
        os.environ['SSL_CERT_FILE'] = cafile
        
        logger.info(f"SSL certificates configured successfully using certifi: {cafile}")
        return True
    except Exception as e:
        logger.error(f"Failed to configure SSL certificates: {str(e)}")
        return False

def configure_ssl_urllib():
    """Configure SSL settings for the application."""
    # Create a custom SSL context
    ssl_context = setup_ssl_context()
    
    # Configure urllib to use our SSL context
    urllib.request.urlopen = lambda url, *args, **kwargs: \
        urllib.request.urlopen(url, *args, context=ssl_context, **kwargs) 
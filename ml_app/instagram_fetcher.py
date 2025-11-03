"""
Instagram Profile Data Fetcher
Fetches Instagram profile information from profile URLs using Apify API
"""

import re
import os
from typing import Dict, Optional

# Try to load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional - environment variables can be set directly
    pass

try:
    from apify_client import ApifyClient

    HAS_APIFY = True
except ImportError:
    HAS_APIFY = False
    print("Apify client not available. Install with: pip install apify-client")


def extract_username_from_url(url: str) -> Optional[str]:
    """Extract username from Instagram URL"""
    # Handle various Instagram URL formats
    patterns = [
        r"instagram\.com/([^/?]+)",
        r"instagram\.com/p/([^/?]+)",
        r"instagram\.com/reel/([^/?]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            username = match.group(1)
            # Skip if it's a post/reel ID (usually long alphanumeric)
            if len(username) > 30 or "/" in username:
                continue
            return username

    return None


def fetch_profile_data_apify(username: str) -> Optional[Dict]:
    """
    Fetch Instagram profile data using Apify API

    Args:
        username: Instagram username (without @)

    Returns:
        Dictionary with profile data or None
    """
    if not HAS_APIFY:
        return None

    try:
        # Get API token from environment variable
        api_token = os.getenv("APIFY_API_TOKEN") or os.getenv("APIFY_TOKEN")

        if not api_token:
            print("Error: APIFY_API_TOKEN not set. Please set your Apify API token.")
            return None

        # Initialize Apify client
        client = ApifyClient(api_token)

        # Prepare the Actor input
        run_input = {
            "usernames": [username],
            "includeAboutSection": True,
        }

        # Run the Actor
        print(f"Fetching profile data for @{username} using Apify...")
        run = client.actor("dSCLg0C3YEZ83HzYX").call(run_input=run_input)

        # Fetch results
        results = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            results.append(item)

        if not results:
            return None

        # Parse Apify result format
        profile = results[0]

        # Map Apify fields to our format
        # Based on actual Apify response structure
        profile_username = profile.get("username", username)
        biography = profile.get("biography", "") or profile.get("bio", "") or ""

        return {
            "username": profile_username,
            "user_media_count": profile.get("postsCount", 0)
            or profile.get("mediaCount", 0)
            or 0,
            "user_follower_count": profile.get("followersCount", 0)
            or profile.get("followers", 0)
            or 0,
            "user_following_count": profile.get("followsCount", 0)
            or profile.get("followingCount", 0)
            or profile.get("following", 0)
            or 0,
            "user_has_profil_pic": (
                1
                if profile.get("profilePicUrl") or profile.get("profilePicUrlHd")
                else 0
            ),
            "user_is_private": (
                1
                if profile.get("private", False) or profile.get("isPrivate", False)
                else 0
            ),
            "user_biography_length": len(biography),
            "username_length": len(profile_username),
            "username_digit_count": sum(1 for c in profile_username if c.isdigit()),
        }
    except Exception as e:
        print(f"Error fetching from Apify: {e}")
        import traceback

        traceback.print_exc()
        return None


def fetch_profile_from_url(url: str) -> Dict:
    """
    Main function to fetch Instagram profile data from URL

    Args:
        url: Instagram profile URL (e.g., https://www.instagram.com/username/)

    Returns:
        Dictionary with profile data or error message
    """
    if not url:
        return {"error": "URL is required"}

    # Clean and validate URL
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    # Extract username
    username = extract_username_from_url(url)

    if not username:
        return {
            "error": "Could not extract username from URL. Please provide a valid Instagram profile URL."
        }

    # Fetch profile data using Apify
    profile_data = fetch_profile_data_apify(username)

    if not profile_data:
        error_msg = f"Could not fetch profile data for @{username}."
        if not HAS_APIFY:
            error_msg += (
                " Apify client is not installed. Install with: pip install apify-client"
            )
        elif not os.getenv("APIFY_API_TOKEN") and not os.getenv("APIFY_TOKEN"):
            error_msg += " Please set APIFY_API_TOKEN environment variable."
        else:
            error_msg += " The profile might be private or does not exist."
        return {"error": error_msg}

    return {"success": True, "username": username, "data": profile_data}


def fetch_profile_from_username(username: str) -> Dict:
    """
    Fetch profile data from username directly

    Args:
        username: Instagram username (without @)

    Returns:
        Dictionary with profile data or error message
    """
    if not username:
        return {"error": "Username is required"}

    username = username.strip().lstrip("@")

    # Fetch profile data using Apify
    profile_data = fetch_profile_data_apify(username)

    if not profile_data:
        error_msg = f"Could not fetch profile data for @{username}."
        if not HAS_APIFY:
            error_msg += (
                " Apify client is not installed. Install with: pip install apify-client"
            )
        elif not os.getenv("APIFY_API_TOKEN") and not os.getenv("APIFY_TOKEN"):
            error_msg += " Please set APIFY_API_TOKEN environment variable."
        else:
            error_msg += " The profile might be private or does not exist."
        return {"error": error_msg}

    return {"success": True, "username": username, "data": profile_data}


if __name__ == "__main__":
    # Test the fetcher
    import json

    test_urls = [
        "https://www.instagram.com/cristiano/",
        "https://www.instagram.com/instagram/",
    ]

    for url in test_urls:
        print(f"\nTesting: {url}")
        result = fetch_profile_from_url(url)
        print(json.dumps(result, indent=2))

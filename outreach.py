import requests
from bs4 import BeautifulSoup
import re
import sqlite3
import googlesearch
import random
import time
from urllib.parse import urlparse

# Proxies and User-Agents to avoid detection
PROXIES = [
    'http://proxy1.com',
    'http://proxy2.com',
    'http://proxy3.com'
]

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'
]

# Set up SQLite database to store emails, names, and websites
conn = sqlite3.connect('leads.db')
cursor = conn.cursor()

# Create a table with fields for first name, email, and website
cursor.execute('''
CREATE TABLE IF NOT EXISTS leads (
    email TEXT PRIMARY KEY,
    first_name TEXT,
    website TEXT
)
''')

# Function to get random user-agent and proxy
def get_random_user_agent():
    return random.choice(USER_AGENTS)

def get_random_proxy():
    return {'http': random.choice(PROXIES)}

# Google Search API function to find leads
def google_search_leads(industry, num_results=10):
    query = f"{industry} contact email"
    search_results = []
    
    for url in googlesearch.search(query, num_results=num_results, stop=num_results, lang='en'):
        search_results.append(url)
    
    return search_results

# LinkedIn search using Google Dorks (or use LinkedIn API if available)
def linkedin_search_leads(industry, num_results=10):
    query = f"site:linkedin.com {industry} contact email"
    search_results = []
    
    # Use Google Search for LinkedIn profiles related to the industry
    for url in googlesearch.search(query, num_results=num_results, stop=num_results, lang='en'):
        if 'linkedin.com' in url:
            search_results.append(url)
    
    return search_results

# ZoomInfo scraping using Google Dorks or API
def zoominfo_search_leads(industry, num_results=10):
    query = f"site:zoominfo.com {industry} contact email"
    search_results = []
    
    for url in googlesearch.search(query, num_results=num_results, stop=num_results, lang='en'):
        if 'zoominfo.com' in url:
            search_results.append(url)
    
    return search_results

# Crunchbase scraping using Google Dorks or API
def crunchbase_search_leads(industry, num_results=10):
    query = f"site:crunchbase.com {industry} contact email"
    search_results = []
    
    for url in googlesearch.search(query, num_results=num_results, stop=num_results, lang='en'):
        if 'crunchbase.com' in url:
            search_results.append(url)
    
    return search_results

# Scrape leads from a given URL with rate limiting and bot detection handling
def scrape_leads_from_url(url):
    try:
        # Add rate limiting (1-3 seconds delay) to avoid bot detection
        time.sleep(random.uniform(1, 3))
        
        # Rotate proxies and user agents to avoid getting blocked
        headers = {'User-Agent': get_random_user_agent()}
        proxies = get_random_proxy()

        response = requests.get(url, headers=headers, proxies=proxies)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract email, first name, and website from the webpage
        email = extract_email(soup)
        first_name = extract_first_name_from_email(email) if email else extract_first_name(soup)
        website = extract_website(soup, url)

        if email and not lead_exists(email):
            # Insert the new lead into the database
            cursor.execute('INSERT INTO leads (email, first_name, website) VALUES (?, ?, ?)',
                           (email, first_name, website))
            conn.commit()
            
            return {'email': email, 'first_name': first_name, 'website': website}
        return None

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# Check if lead already exists in the database
def lead_exists(email):
    cursor.execute('SELECT * FROM leads WHERE email = ?', (email,))
    return cursor.fetchone() is not None

# Extract first name from email using a heuristic
def extract_first_name_from_email(email):
    if not email:
        return None
    username = email.split('@')[0]  # Get the part before @
    parts = re.split(r'[._-]', username)
    
    if parts and len(parts[0]) > 1:
        first_name = parts[0].capitalize()  # Capitalize the first name
        return first_name
    return "User"  # Edge case fallback

# Extract first name from the webpage (fallback)
def extract_first_name(soup):
    possible_names = soup.find_all(['meta', 'h1', 'p', 'h2'])
    for tag in possible_names:
        if tag.get('content') and len(tag.get('content').split()) == 2:
            return tag.get('content').split()[0]
        if tag.text and len(tag.text.split()) == 2:
            return tag.text.split()[0]
    return "User"  # Fallback if no name found

# Extract email using a broader search
def extract_email(soup):
    potential_emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', soup.text)
    if potential_emails:
        email = potential_emails[0]
        if is_valid_email(email):
            return email
    return None

# Validate email address format
def is_valid_email(email):
    return bool(re.match(r'[^@]+@[^@]+\.[^@]+', email))

# Extract website from meta tags or hrefs
def extract_website(soup, url):
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    meta_tags = soup.find_all('meta', attrs={'property': 'og:url'})
    if meta_tags:
        return meta_tags[0].get('content')
    
    links = soup.find_all('a', href=True)
    for link in links:
        href = link.get('href')
        if 'http' in href and base_url in href:
            return href
    return base_url  # Return the base URL if no specific website found

# Handle edge cases in scraping
def handle_edge_cases(soup, url):
    if not soup:
        print(f"Could not parse {url}")
        return None
    
    email = extract_email(soup)
    if not email:
        print(f"No valid email found on {url}")
        return None
    
    first_name = extract_first_name_from_email(email) if email else "User"
    website = extract_website(soup, url) if soup else "Unknown"
    
    return {'email': email, 'first_name': first_name, 'website': website}

# Main function to find and email leads based on industry
def find_and_email_leads(industry, num_leads=10):
    # Collect leads from various sources
    google_leads = google_search_leads(industry, num_results=num_leads * 2)
    linkedin_leads = linkedin_search_leads(industry, num_results=num_leads)
    zoominfo_leads = zoominfo_search_leads(industry, num_results=num_leads)
    crunchbase_leads = crunchbase_search_leads(industry, num_results=num_leads)

    all_leads = google_leads + linkedin_leads + zoominfo_leads + crunchbase_leads

    found_leads = []

    for url in all_leads:
        lead = scrape_leads_from_url(url) or handle_edge_cases(BeautifulSoup(requests.get(url).text, 'html.parser'), url)
        if lead:
            found_leads.append(lead)
            if len(found_leads) >= num_leads:
                break

    # Send emails to the found leads
    for lead in found_leads:
        send_email(lead['email'], lead['first_name'], lead['website'], industry)

# Function to send personalized email (placeholder)
def send_email(email, first_name, website, industry):
    print(f"Sending email to: {email} ({first_name}) about {industry}")
    print(f"Website: {website}")

# Example usage
find_and_email_leads('luxury real estate')

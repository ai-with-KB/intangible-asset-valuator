import requests
import pandas as pd

# Function to fetch patent data from OpenAlex
def fetch_openalex_patents(keyword="AI", size=10):
    url = f"https://api.openalex.org/works?search={keyword}&page=1&per_page={size}"
    
    # Print the full request URL for debugging
    print("Request URL:", url)

    response = requests.get(url)
    
    # Print the full response to debug
    print("Response Status Code:", response.status_code)
    print("Response JSON:", response.json())  # This will show the full response from OpenAlex
    
    if response.status_code == 200:
        data = response.json()
        docs = data.get("results", [])
        
        # Create a DataFrame from the JSON response
        df = pd.json_normalize(docs)
        
        # Save the data as a CSV file
        df.to_csv("openalex_patents.csv", index=False)
        print(f"✅ Fetched and saved {len(df)} works.")
    else:
        print("❌ Failed to fetch:", response.status_code, response.text)

if __name__ == "__main__":
    fetch_openalex_patents(keyword="ai and finance")  # Change the keyword if needed

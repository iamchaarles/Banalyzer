import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import quote
import json

def download_bing_images(query, max_images=500, output_dir="banana_images"):
    """Download images from Bing search with proper pagination"""
    
    # Create output directory
    safe_query = query.replace(" ", "_")
    full_path = os.path.join(output_dir, safe_query)
    os.makedirs(full_path, exist_ok=True)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.bing.com/'
    }
    
    downloaded = 0
    first = 1  # Bing uses 'first' parameter for pagination (1, 101, 201, etc.)
    consecutive_failures = 0
    max_failures = 3
    
    print(f"Downloading images for: {query}")
    print("-" * 60)
    
    while downloaded < max_images and consecutive_failures < max_failures:
        # Bing pagination: first=1, first=101, first=201, etc (increments by ~100)
        url = f"https://www.bing.com/images/async?q={quote(query)}&first={first}&count=100&mmasync=1"
        
        print(f"\nFetching page (offset: {first})...")
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                print(f"Failed to fetch page: HTTP {response.status_code}")
                consecutive_failures += 1
                first += 100
                time.sleep(2)
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all image container links
            image_links = soup.find_all('a', {'class': 'iusc'})
            
            if not image_links:
                print(f"No images found on this page")
                consecutive_failures += 1
                first += 100
                time.sleep(2)
                continue
            
            print(f"Found {len(image_links)} images on this page")
            consecutive_failures = 0  # Reset on success
            
            for link in image_links:
                if downloaded >= max_images:
                    break
                
                try:
                    # Extract the 'm' attribute which contains JSON with image URL
                    m_attr = link.get('m')
                    
                    if m_attr:
                        img_data = json.loads(m_attr)
                        img_url = img_data.get('murl')  # 'murl' is the medium-quality image URL
                        
                        if img_url:
                            # Download the image
                            img_response = requests.get(img_url, headers=headers, timeout=10, stream=True)
                            
                            if img_response.status_code == 200:
                                # Get file extension from URL or content-type
                                content_type = img_response.headers.get('content-type', '')
                                
                                if 'jpeg' in content_type or img_url.endswith(('.jpg', '.jpeg')):
                                    ext = 'jpg'
                                elif 'png' in content_type or img_url.endswith('.png'):
                                    ext = 'png'
                                elif 'webp' in content_type or img_url.endswith('.webp'):
                                    ext = 'webp'
                                else:
                                    ext = 'jpg'  # default
                                
                                filename = f"{safe_query}_{downloaded:04d}.{ext}"
                                filepath = os.path.join(full_path, filename)
                                
                                # Save the image
                                with open(filepath, 'wb') as f:
                                    for chunk in img_response.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                
                                downloaded += 1
                                print(f"  ✓ Downloaded {downloaded}/{max_images}: {filename}")
                                
                                # Small delay between downloads
                                time.sleep(0.3)
                            else:
                                print(f"  ✗ Failed to download image: HTTP {img_response.status_code}")
                
                except json.JSONDecodeError:
                    print(f"  ✗ Failed to parse image metadata")
                    continue
                except requests.exceptions.RequestException as e:
                    print(f"  ✗ Download error: {e}")
                    continue
                except Exception as e:
                    print(f"  ✗ Unexpected error: {e}")
                    continue
            
            # Move to next page (Bing increments by ~100)
            first += 100
            
            # Delay between page requests (be respectful)
            time.sleep(2)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page: {e}")
            consecutive_failures += 1
            first += 100
            time.sleep(3)
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            consecutive_failures += 1
            first += 100
            time.sleep(3)
            continue
    
    print("\n" + "=" * 60)
    print(f"Download complete! Downloaded {downloaded} images")
    print(f"Saved to: {full_path}")
    print("=" * 60)
    
    return downloaded


# Usage example
if __name__ == "__main__":
    print("Starting bulk banana image download...\n")
    
    categories = [
        ("green banana", 500),
        #("ripe yellow banana", 300),
        #("overripe banana brown spots", 300),
        #("rotten black banana", 300)
    ]
    
    for query, limit in categories:
        print(f"\n{'='*60}")
        print(f"Category: {query.upper()}")
        print(f"{'='*60}")
        download_bing_images(query, max_images=limit, output_dir="banana_images")
        time.sleep(3)  # Delay between different searches
    
    print("\n🎉 All downloads complete!")

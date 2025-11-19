from bing_image_downloader import downloader

# Download images for different banana categories
# Adjust 'limit' based on how many images you need per category

print("Starting bulk download...\n")
'''
# Ripe bananas
print("Downloading RIPE bananas...")
downloader.download(
    query="ripe yellow banana",
    limit=300,
    output_dir="banana_images",
    adult_filter_off=True,
    force_replace=False,
    timeout=60,
    verbose=True
)
'''
# Overripe bananas
'''print("\nDownloading OVERRIPE bananas...")

downloader.download(
    query="overripe banana brown spots",
    limit=300,
    output_dir="banana_images",
    adult_filter_off=True,
    force_replace=False,
    timeout=60,
    verbose=True
)'''

# Rotten bananas
'''print("\nDownloading ROTTEN bananas...")
downloader.download(
    query="rotten black banana",
    limit=300,
    output_dir="banana_images",
    adult_filter_off=True,
    force_replace=False,
    timeout=60,
    verbose=True
)'''

# Unripe bananas
print("\nDownloading UNRIPE green bananas...")
downloader.download(
    query="green banana",
    limit=500,
    output_dir="banana_images",
    adult_filter_off=True,
    force_replace=False,
    timeout=60,
    verbose=True
)

print("\n" + "="*60)
print("Download complete!")
print("Check the 'banana_images' folder for your images")
print("="*60)
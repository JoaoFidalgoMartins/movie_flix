import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from typing import List
import aiohttp
import asyncio
import nest_asyncio
import numpy as np
from visualizer.config import CONFIG

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()


class RecommendationCarousel:
    def __init__(self, title: str, recommended_items: List[int]):
        self.title = title
        self.recommended_items = recommended_items
        self.recommended_items_urls = []

class Visualizer:
    def __init__(self):
        self.map_movie_title = {}
        self.map_movie_link = {}
        self.ratings = None
        self.tmdb_api_key = CONFIG['tmdb_api_key']

    def load(self, df_movies, df_links, df_ratings):
        """Load movie data mappings."""
        self.map_movie_title = df_movies.set_index('movie_id')['title'].to_dict()
        self.map_movie_link = (
            df_links.set_index('movie_id')['tmdb_id'].fillna(-1).astype(int).to_dict()
        )
        self.ratings = df_ratings

    def visualize_recommendation_carousels(self, carousels: List[RecommendationCarousel], n_cols=5):
        """Display a grid of recommended movies."""
        for carousel in carousels:
            self.get_urls(carousel)
        self.display_carousels(carousels, n_cols)

    def get_urls(self, carousel: RecommendationCarousel):
        """Retrieve poster URLs for recommended movies."""
        carousel.recommended_items_urls = asyncio.run(
            self.fetch_movie_poster_urls(carousel.recommended_items)
        )

    def display_carousels(self, carousels: List[RecommendationCarousel], n_cols: int):
        """Display images in a grid format."""
        n_rows = len(carousels)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = np.atleast_2d(axes)  # Ensure axes is always 2D

        for row_idx, carousel in enumerate(carousels):
            axes[row_idx, 0].set_title(carousel.title, fontsize=14, fontweight='bold', loc='left', pad=20)
            posters = asyncio.run(self.fetch_all_posters(carousel.recommended_items_urls[:n_cols]))

            for col_idx, poster in enumerate(posters):
                ax = axes[row_idx, col_idx]
                ax.imshow(Image.open(BytesIO(poster)))
                ax.axis('off')

            for col_idx in range(len(posters), n_cols):
                axes[row_idx, col_idx].axis('off')

        plt.show()

    async def fetch_movie_poster_urls(self, movie_ids: List[int]) -> List[str]:
        """Fetch movie poster URLs from TMDb API."""
        urls = []
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_movie_poster_url(session, movie_id) for movie_id in movie_ids]
            urls = await asyncio.gather(*tasks)
        return urls

    async def fetch_movie_poster_url(self, session: aiohttp.ClientSession, movie_id: int) -> str:
        """Fetch a single movie poster URL."""
        tmdb_id = self.map_movie_link.get(movie_id, -1)
        if tmdb_id == -1:
            return None

        url = f'https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={self.tmdb_api_key}'
        async with session.get(url) as response:
            data = await response.json()
            poster_path = data.get('poster_path')
            return f'https://image.tmdb.org/t/p/w500{poster_path}' if poster_path else None

    async def fetch_all_posters(self, urls: List[str]) -> List[bytes]:
        """Download all poster images asynchronously."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_poster(session, url) for url in urls if url]
            return await asyncio.gather(*tasks)

    async def fetch_poster(self, session: aiohttp.ClientSession, url: str) -> bytes:
        """Fetch a single poster image."""
        async with session.get(url) as response:
            return await response.read()

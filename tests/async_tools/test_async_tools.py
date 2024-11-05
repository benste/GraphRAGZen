import asyncio
import unittest
from unittest.mock import AsyncMock

from graphragzen.async_tools import async_loop


class TestAsyncLoop(unittest.TestCase):
    async def test_async_loop_basic(self):
        async def mock_func(item):
            return item * 2

        items = [1, 2, 3]
        results = await async_loop(mock_func, items)
        self.assertEqual(results, [2, 4, 6])

    async def test_async_loop_empty_list(self):
        async def mock_func(item):
            return item * 2

        items = []
        results = await async_loop(mock_func, items)
        self.assertEqual(results, [])

    async def test_async_loop_with_args_and_kwargs(self):
        async def mock_func(item, multiplier, adder):
            return item * multiplier + adder

        items = [1, 2, 3]
        results = await async_loop(mock_func, items, multiplier=2, adder=1)
        self.assertEqual(results, [3, 5, 7])

    async def test_async_loop_with_exception(self):
        async def mock_func(item):
            if item == 2:
                raise ValueError("Test exception")
            return item * 2

        items = [1, 2, 3]
        with self.assertRaises(ValueError):
            await async_loop(mock_func, items)

    async def test_async_loop_with_tqdm(self):
        async def mock_func(item):
            await asyncio.sleep(0.1)
            return item * 2

        items = [1, 2, 3, 4, 5]
        results = await async_loop(mock_func, items, loop_description="Testing tqdm")
        self.assertEqual(results, [2, 4, 6, 8, 10])

    async def test_async_loop_with_non_async_func(self):
        def sync_func(item):
            return item * 2

        items = [1, 2, 3]
        with self.assertRaises(TypeError):
            await async_loop(sync_func, items)


if __name__ == "__main__":
    unittest.main()
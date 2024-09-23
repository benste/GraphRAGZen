from typing import Any, Callable

from tqdm.asyncio import tqdm


async def async_loop(
    func: Callable, items: list, loop_description: str = "", *args: Any, **kwargs: Any
) -> Any:
    """Call a function for each item in items asynchronously

    note that *args and **kwargs to add to the func call can be supplied after the input specified
    below.

    Args:
        func (Callable): Function to call
        items (list): List of items for which to call func individually
        loop_description (str, optional): Description of the loop, used for tqdm. Defaults to "".

    Returns:
        _type_: _description_
    """
    tasks = [func(item, *args, **kwargs) for item in items]

    # Gather all the tasks and run them concurrently
    # results = await asyncio.gather(*tasks)
    results = await tqdm.gather(*tasks, desc=loop_description)
    return results

from typing import Any, Generator
from typing_extensions import Self
from collections.abc import Mapping

from pydantic import BaseModel


class MappedBaseModel(BaseModel, Mapping):
    """Extension of pydantic BaseModel to allow unpacking

    example:
    ```
    class Car(MappedBaseModel):
        num_wheels: int
        horsepower: int
        free_text: str

    def describe_car(**kwargs):
        print(kwargs)

    my_car = Car(
        num_wheels = 3,
        horsepower = 10,
        free_text = "Unique car with single front wheel in the center",
    )
    describe_car(**my_car)
    ```

    """

    def __init__(self: Self, /, **data: Any) -> None:
        # Handle:
        # 1. dict input: data = {arg1: value1, arg2: value2, arg3:...}
        # 2. self input: data = self
        # 3. self in dict input: data = {arg1: value1, arg2: self}

        # Find values in data that are already this class (case 2 and 3) and
        # merge with the rest of the dict
        merged_data: dict = {}
        for key, value in data.items():
            if isinstance(value, self.__class__):
                # Merge with rest of data, giving non-pydantic class keys preference
                merged_data = value.__dict__ | merged_data
            else:
                merged_data[key] = value

        __tracebackhide__ = True
        self.__pydantic_validator__.validate_python(merged_data, self_instance=self)

    def __getitem__(self: Self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self: Self) -> Generator:
        return iter(self.__dict__)  # type: ignore

    def __len__(self: Self) -> int:
        return len(self.__dict__)

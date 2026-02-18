"""Primitives that require the _jac JS runtime object.

Operations here compile to _jac.list.sort(), _jac.list.count(), etc.
These test that the _jac runtime is properly provided at execution time.
"""

def jac_runtime_py() -> dict {
    # list.sort with key function
    nums: list = [3, 1, 4, 1, 5, 9, 2, 6];
    nums.sort(key=lambda x: -x);

    # list.count
    items: list = [1, 2, 2, 3, 3, 3];
    count_2: int = items.count(2);
    count_3: int = items.count(3);

    # list.remove
    removable: list = [10, 20, 30, 20];
    removable.remove(20);

    # str.capitalize
    cap: str = "hello world".capitalize();

    return {
        "sort_desc": nums,
        "count_2": count_2,
        "count_3": count_3,
        "after_remove": removable,
        "capitalize": cap
    };
}

cl def jac_runtime_cl() -> dict {
    # list.sort with key function
    nums: list = [3, 1, 4, 1, 5, 9, 2, 6];
    nums.sort(key=lambda x: -x);

    # list.count
    items: list = [1, 2, 2, 3, 3, 3];
    count_2: int = items.count(2);
    count_3: int = items.count(3);

    # list.remove
    removable: list = [10, 20, 30, 20];
    removable.remove(20);

    # str.capitalize
    cap: str = "hello world".capitalize();

    return {
        "sort_desc": nums,
        "count_2": count_2,
        "count_3": count_3,
        "after_remove": removable,
        "capitalize": cap
    };
}

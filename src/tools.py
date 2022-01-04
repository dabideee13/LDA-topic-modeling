# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, Any
import json


def get_data_path(file: Optional[str] = None) -> Path:
    return Path.joinpath(Path.cwd(), "data", file)


def read_data(file: str) -> None:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def export_data(file: str, data: Any) -> None:
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def pipe(raw_input: Any, *functions, **functions_with_args) -> Any:
    output = raw_input

    if functions:
        for function in functions:
            output = function(output)

    if functions_with_args:
        for function, args_list in functions_with_args.items():
            output = eval(function)(output, *args_list)

    return output
import os
from typing import Any

import pytest

from mytriton.cuda_utils import CudaUnavailableError, cuda_module


@pytest.fixture(scope="session")
def cp() -> Any:
    require_cuda = os.environ.get("MYTRITON_REQUIRE_CUDA") == "1"
    try:
        return cuda_module()
    except CudaUnavailableError as error:
        if require_cuda:
            pytest.fail(str(error))
        pytest.skip(str(error))

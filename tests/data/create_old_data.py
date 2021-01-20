import pickle

import torch
import torchkbnufft as tkbn
import sys

sys.path.append("../")

from conftest import create_input_plus_noise, create_ktraj


def create_interp_data():
    torch.set_default_dtype(torch.double)
    test_params = [
        # ([3, 1, 38, 2], 52, False),
        ([5, 1, 78, 96, 2], 410, False),
        ([1, 2, 17, 19, 12, 2], 112, False),
    ]

    outputs = []
    for (shape, klength, is_complex) in test_params:
        torch.manual_seed(123)
        im_size = shape[2:-1]

        image = create_input_plus_noise(shape, is_complex)
        if image.ndim == 4:
            dim = (2,)
        elif image.ndim == 5:
            dim = (2, 3)
        elif image.ndim == 6:
            dim = (2, 3, 4)
        image = torch.view_as_real(
            torch.fft.fftn(torch.view_as_complex(image), dim=dim)
        )
        if image.ndim == 4:
            image = image.permute(0, 1, 3, 2).contiguous()
        elif image.ndim == 5:
            image = image.permute(0, 1, 4, 2, 3).contiguous()
        elif image.ndim == 6:
            image = image.permute(0, 1, 5, 2, 3, 4).contiguous()

        ktraj = (
            create_ktraj(len(im_size), klength)
            .unsqueeze(0)
            .repeat(image.shape[0], 1, 1)
        )

        forw_ob = tkbn.KbInterpForw(im_size=im_size, grid_size=im_size)

        kdata = forw_ob(image, ktraj)

        if image.ndim == 4:
            image = image.permute(0, 1, 3, 2).contiguous()
        elif image.ndim == 5:
            image = image.permute(0, 1, 3, 4, 2).contiguous()
        elif image.ndim == 6:
            image = image.permute(0, 1, 3, 4, 5, 2).contiguous()
        kdata = kdata.permute(0, 1, 3, 2).contiguous()

        outputs.append((image, ktraj[0], kdata))

    with open("interp_data.pkl", "wb") as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    create_interp_data()

from typing import Optional

import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def draw3d_arrow(arrow_location: np.ndarray,
                 arrow_vector: np.ndarray,
                 head_length: float = 0.3,
                 alpha: float = 1.0,
                 color: Optional[str] = None,
                 name: Optional[str] = None,
                 ax: Optional[Axes3D] = None, ) -> Axes3D:
    if ax is None:
        ax = plt.gca(projection="3d")

    ax.quiver(
        *arrow_location,
        *arrow_vector,
        arrow_length_ratio=head_length / np.linalg.norm(arrow_vector),
        color=color, alpha=alpha
    )
    if name is not None:
        ax.text(*(arrow_location + arrow_vector), name)

    return ax


class ReferenceFrame:
    def __init__(
            self,
            origin: np.ndarray,
            dx: np.ndarray,
            dy: np.ndarray,
            dz: np.ndarray,
            name: str,
    ) -> None:
        self.origin = origin
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.name = name

    def draw3d(
            self,
            head_length: float = 0.3,
            color: str = "tab:blue",
            ax: Optional[Axes3D] = None,
    ) -> Axes3D:
        if ax is None:
            ax = plt.gca(projection="3d")

        ax.text(*self.origin + 0.5, f"({self.name})")
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dx,
            head_length=head_length,
            color="red",  # color,
            name="x",
        )
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dy,
            head_length=head_length,
            color="green",  # color,
            name="y",
        )
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dz,
            head_length=head_length,
            color=color,
            name="z",
        )
        return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # World coordinate, Camera coordinate axis ##
    world_origin = np.zeros(3)
    dx, dy, dz = np.eye(3)
    t = np.array([3, -4, 2])
    world_frame = ReferenceFrame(
        origin=world_origin,
        dx=dx,
        dy=dy,
        dz=dz,
        name="World",
    )
    camera_frame = ReferenceFrame(
        origin=t,
        dx=dx,
        dy=dy,
        dz=dz,
        name="Camera",
    )
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    world_frame.draw3d()
    camera_frame.draw3d()
    draw3d_arrow(world_origin, t, color="tab:red", name="t")
    ax.set_xlim3d([-3, 3])
    ax.set_zlim3d([-3, 3])
    ax.set_ylim3d([-3, 3])
    ax.set_title(f"Camera Translation (t = {t})")
    plt.tight_layout()
    plt.show()

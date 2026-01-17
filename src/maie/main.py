import random

from maie.playground import Playground2D
from maie.worlds.wfc_world import WfcWorld, WfcWorldConfig
from maie.worlds.uuworld import UUWorld, UUWorldConfig
from maie.worlds.poisson_world import PoissonWorld, PoissonWorldConfig
from maie.worlds.myworld import MyWorld, MyWorldConfig


def main():
    # world = UUWorld(UUWorldConfig())
    # world = PoissonWorld(PoissonWorldConfig())
    # cfg = WfcWorldConfig()
    # cfg.tileset = "city"
    # world = WfcWorld(cfg)
    
    # Factory function to create new worlds (used for regeneration with 'R' key)
    def create_world():
        cfg = MyWorldConfig()
        cfg.seed = random.randint(0, 2**31 - 1)
        print(f"Using seed: {cfg.seed}")
        return MyWorld(cfg)
    
    world = create_world()
    pg = Playground2D(world, world_factory=create_world)
    pg.run()


if __name__ == "__main__":
    main()

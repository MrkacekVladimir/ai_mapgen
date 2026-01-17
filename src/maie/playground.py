import math
import sys
from datetime import datetime
from typing import Callable
from typing import Protocol

import pygame

from maie.camera import Camera2D, RenderContext, DrawLayer, InputState
from maie.common import Vec2, GVec2


class CfgLike(Protocol):
    width: int
    height: int
    tile_size: float


class TeamCfgLike(Protocol):
    """Protocol for configs that support team configuration."""
    teams: list[int]


class WorldLike(Protocol):
    cfg: CfgLike

    def get_layers(self, layers: list[int]) -> list[DrawLayer]:
        ...

    def debug_layers(self) -> list[DrawLayer]:
        ...

pygame.init()

HUD_FONT = pygame.font.SysFont("consolas", 16)
HUD_FONT_COLOR_2 = (160, 160, 160)
HUD_FONT_COLOR = (220, 220, 220)

FILL_COLOR = (18, 18, 18)
COLOR_AXES = (90, 90, 90)

# Layer names for UI panel (indices 0-9 are regular layers, 10-11 are debug)
LAYER_NAMES = [
    "Border",
    "Terrain",
    "Forests",
    "Rivers",
    "Cliffs",
    "Trails",
    "Spawn Zones",
    "Spawns",
    "Gold Mines",
    "Creep Camps",
]
COLOR_LAYER_ON = (100, 220, 100)
COLOR_LAYER_OFF = (100, 100, 100)


class Playground2D:
    def __init__(
        self,
        world: WorldLike,
        w: int = 1920,
        h: int = 1080,
        name="uuWorld",
        world_factory: Callable[[], WorldLike] | None = None,
        world_config: TeamCfgLike | None = None,
    ) -> None:
        pygame.display.set_caption(name)
        self.screen = pygame.display.set_mode((w, h))
        self.clock = pygame.time.Clock()
        self.name = name

        self.cam = Camera2D(offset=pygame.Vector2(0, 0), zoom=1.0)
        self.world = world
        self.world_factory = world_factory
        self.world_config = world_config  # Shared config for team adjustments
        self.layers_to_draw = list(range(10))  # All non-debug layers on by default

        self.tile_size = world.cfg.tile_size
        self.show_grid = True
        self.show_axes = True
        self.debug = False
        self.debug_layers = self.world.debug_layers()
        self.current_layer = 0

        self._panning = False
        self._pan_anchor = pygame.Vector2(0, 0)
        self._cam_anchor = pygame.Vector2(0, 0)

        self.on_frame: Callable | None = None


    def tile_at_world(self, p: Vec2) -> GVec2:
        return int(math.floor(p[0] / self.tile_size)), int(math.floor(p[1] / self.tile_size))

    def _draw_grid(self, ctx: RenderContext) -> None:
        w, h = ctx.screen.get_size()

        # visible world bounds
        top_left = ctx.camera.screen_to_world((0, 0))
        bottom_right = ctx.camera.screen_to_world((w, h))

        x0, y0 = top_left
        x1, y1 = bottom_right
        ts = self.tile_size

        # grid line thickness adapts a bit
        width = 1 if ctx.camera.zoom >= 0.5 else 1

        # vertical lines
        gx0 = int(math.floor(min(x0, x1) / ts)) - 1
        gx1 = int(math.floor(max(x0, x1) / ts)) + 1
        gy0 = int(math.floor(min(y0, y1) / ts)) - 1
        gy1 = int(math.floor(max(y0, y1) / ts)) + 1

        grid_col = (40, 40, 40)

        for gx in range(gx0, gx1 + 1):
            x = gx * ts
            a = ctx.camera.world_to_screen((x, gy0 * ts))
            b = ctx.camera.world_to_screen((x, (gy1 + 1) * ts))
            pygame.draw.line(ctx.screen, grid_col, a, b, width)

        for gy in range(gy0, gy1 + 1):
            y = gy * ts
            a = ctx.camera.world_to_screen((gx0 * ts, y))
            b = ctx.camera.world_to_screen(((gx1 + 1) * ts, y))
            pygame.draw.line(ctx.screen, grid_col, a, b, width)

    def _draw_axes(self, ctx: RenderContext) -> None:
        w, h = ctx.screen.get_size()
        origin = ctx.camera.world_to_screen((0, 0))
        pygame.draw.line(ctx.screen, COLOR_AXES, (0, origin[1]), (w, origin[1]), 1)  # x-axis
        pygame.draw.line(ctx.screen, COLOR_AXES, (origin[0], 0), (origin[0], h), 1)  # y-axis

    def _draw_hud(self, ctx: RenderContext) -> None:
        wx, wy = ctx.input.mouse_world
        tx, ty = self.tile_at_world((wx, wy))
        mode = "DEBUG" if self.debug else "NORMAL"
        current_layer = "  layer=" + self.debug_layers[self.current_layer].label if self.debug else ""
        text = f"zoom={ctx.camera.zoom:.3f}  world=({wx:.1f},{wy:.1f})  tile=({tx},{ty})  mode={mode}{current_layer}"
        surf = HUD_FONT.render(text, True, HUD_FONT_COLOR)
        ctx.screen.blit(surf, (10, 10))

        # Team configuration line
        if self.world_config is not None:
            teams = self.world_config.teams
            num_teams = len(teams)
            players_per_team = teams[0] if teams else 1
            total_players = sum(teams)
            team_text = f"Teams: {num_teams}x{players_per_team} ({total_players} players) | T/Shift+T: teams | P/Shift+P: players | R: regenerate"
            surf_teams = HUD_FONT.render(team_text, True, HUD_FONT_COLOR)
            ctx.screen.blit(surf_teams, (10, 30))
            help_y = 50
        else:
            help_y = 30

        help1 = "LMB drag: pan | Wheel: zoom | G: grid | A: axes | D: debug mode | TAB: cycle layers | S: save | ESC: quit"
        surf2 = HUD_FONT.render(help1, True, HUD_FONT_COLOR_2)
        ctx.screen.blit(surf2, (10, help_y))

        # Right panel - Layer status
        self._draw_layer_panel(ctx)

    def _draw_layer_panel(self, ctx: RenderContext) -> None:
        """Draw the layer status panel on the right side of the screen."""
        screen_w, screen_h = ctx.screen.get_size()
        panel_x = screen_w - 180
        panel_y = 10
        line_height = 22

        # Panel header
        header = HUD_FONT.render("LAYERS", True, HUD_FONT_COLOR)
        ctx.screen.blit(header, (panel_x, panel_y))
        panel_y += line_height

        # Separator line
        pygame.draw.line(
            ctx.screen,
            HUD_FONT_COLOR_2,
            (panel_x, panel_y),
            (panel_x + 160, panel_y),
            1
        )
        panel_y += 8

        # Layer entries
        for idx, name in enumerate(LAYER_NAMES):
            is_on = idx in self.layers_to_draw
            color = COLOR_LAYER_ON if is_on else COLOR_LAYER_OFF
            status = "ON" if is_on else "OFF"
            text = f"[{idx}] {name:<12} {status}"
            surf = HUD_FONT.render(text, True, color)
            ctx.screen.blit(surf, (panel_x, panel_y))
            panel_y += line_height

    # --- events ---
    def _handle_event(self, e: pygame.event.Event) -> None:
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit(0)
            if e.key == pygame.K_g:
                self.show_grid = not self.show_grid
            if e.key == pygame.K_a:
                self.show_axes = not self.show_axes
            if e.key == pygame.K_d:
                self.debug = not self.debug
            if e.key == pygame.K_TAB:
                self.current_layer = (self.current_layer + 1) % len(self.debug_layers)
            if e.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                val = e.key - 48
                if val in self.layers_to_draw:
                    self.layers_to_draw.remove(val)
                else:
                    self.layers_to_draw.append(val)
            if e.key == pygame.K_s:
                self.save()
            if e.key == pygame.K_r:
                self._regenerate_world()
            if e.key == pygame.K_t:
                self._adjust_teams(e)
            if e.key == pygame.K_p:
                self._adjust_players(e)

        if e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == 1:  # MMB pan
                self._panning = True
                self._pan_anchor = pygame.Vector2(e.pos)
                self._cam_anchor = self.cam.offset.copy()

            if e.button == 4:  # wheel up
                self.cam.zoom_at(e.pos, 1.15)
            if e.button == 5:  # wheel down
                self.cam.zoom_at(e.pos, 1 / 1.15)

        if e.type == pygame.MOUSEBUTTONUP:
            if e.button == 1:
                self._panning = False

        if e.type == pygame.MOUSEMOTION and self._panning:
            delta = pygame.Vector2(e.pos) - self._pan_anchor
            self.cam.offset = self._cam_anchor + delta

    def save(self):
        camera = Camera2D(offset=(0, 0), zoom=1.0)
        ms = pygame.mouse.get_pos()
        inp = InputState(
            mouse_world=self.cam.screen_to_world(ms),
            mouse_screen=ms
        )
        ctx = RenderContext(
            screen=pygame.Surface((self.world.cfg.width, self.world.cfg.height)),
            camera=camera,
            input=inp,
            debug=False
        )
        layers = self.world.get_layers(self.layers_to_draw)
        layers.sort(key=lambda x: x.z)
        for layer in layers:
            layer.draw(ctx)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        pygame.image.save(ctx.screen, f"{self.name}_{now}.jpg")

    def _regenerate_world(self) -> None:
        """Regenerate the world using the factory function if available."""
        if self.world_factory is None:
            print("No world factory set - cannot regenerate")
            return
        print("Regenerating world...")
        self.world = self.world_factory()
        self.tile_size = self.world.cfg.tile_size
        self.debug_layers = self.world.debug_layers()
        self.current_layer = 0
        print("World regenerated!")

    def _adjust_teams(self, e: pygame.event.Event) -> None:
        """Adjust number of teams with T (add) / Shift+T (remove)."""
        if self.world_config is None:
            return
        mods = pygame.key.get_mods()
        teams = self.world_config.teams
        players_per_team = teams[0] if teams else 1

        if mods & pygame.KMOD_SHIFT:
            # Remove a team (min 1)
            if len(teams) > 1:
                self.world_config.teams = teams[:-1]
                print(f"Teams: {len(self.world_config.teams)}x{players_per_team}")
        else:
            # Add a team (max 8)
            if len(teams) < 8:
                self.world_config.teams = teams + [players_per_team]
                print(f"Teams: {len(self.world_config.teams)}x{players_per_team}")

    def _adjust_players(self, e: pygame.event.Event) -> None:
        """Adjust players per team with P (add) / Shift+P (remove)."""
        if self.world_config is None:
            return
        mods = pygame.key.get_mods()
        teams = self.world_config.teams
        players_per_team = teams[0] if teams else 1

        if mods & pygame.KMOD_SHIFT:
            # Remove a player per team (min 1)
            if players_per_team > 1:
                new_ppt = players_per_team - 1
                self.world_config.teams = [new_ppt] * len(teams)
                print(f"Teams: {len(teams)}x{new_ppt}")
        else:
            # Add a player per team (max 4)
            if players_per_team < 4:
                new_ppt = players_per_team + 1
                self.world_config.teams = [new_ppt] * len(teams)
                print(f"Teams: {len(teams)}x{new_ppt}")

    def run(self, fps: int = 30) -> None:
        while True:
            for e in pygame.event.get():
                self._handle_event(e)

            if self.on_frame is not None:
                self.on_frame(self)

            ms = pygame.mouse.get_pos()
            inp = InputState(
                mouse_world=self.cam.screen_to_world(ms),
                mouse_screen=ms
            )
            ctx = RenderContext(
                screen=self.screen,
                camera=self.cam,
                input=inp,
                debug=False
            )

            layers = [
                DrawLayer(z=2000, label="hud", draw=self._draw_hud),
            ]
            if self.show_grid:
                layers.append(DrawLayer(z=1000, label="grid", draw=self._draw_grid))
            if self.show_axes:
                layers.append(DrawLayer(z=1001, label="axes", draw=self._draw_axes))

            if self.debug:
                layers.append(self.debug_layers[self.current_layer])
            else:
                layers.extend(self.world.get_layers(self.layers_to_draw))
            layers = sorted(layers, key=lambda x: x.z)

            self.screen.fill(FILL_COLOR)
            for layer in layers:
                layer.draw(ctx)

            pygame.display.flip()
            self.clock.tick(fps)

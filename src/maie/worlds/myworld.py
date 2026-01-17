"""
Warcraft 3-Style Strategy Game World Generator

Generates maps with:
- Radial team spawn points
- Gold mines (starting + neutral)
- Neutral creep camps (easy/medium/hard based on distance from spawns)
- Strategy-game terrain (mostly land)
- Natural trails between teammates (Dijkstra-based)
- Forest coverage using cellular automata
- Team separation features (rivers, cliffs, dense forests)
"""
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Iterable

import numpy as np
import pygame

from maie.camera import RenderContext, DrawLayer, draw_tile
from maie.common import Vec2, Color, GVec2, clamp
from maie.dijkstra import dijkstra_grid, reconstruct_path_grid
from maie.perlin import perlin2d_fbm
from maie.poisson import poisson_disc_2d


# =============================================================================
# Enums
# =============================================================================
class TerrainType(IntEnum):
    """Terrain types for the map."""
    WATER = 0
    LAND = 1
    FOREST = 2
    DENSE_FOREST = 3
    CLIFF = 4
    RIVER = 5
    BRIDGE = 6


class CreepDifficulty(IntEnum):
    EASY = 0
    MEDIUM = 1
    HARD = 2


class ViewLayers(IntEnum):
    BORDER = 0
    TERRAIN = auto()
    FORESTS = auto()
    RIVERS = auto()
    CLIFFS = auto()
    TRAILS = auto()
    SPAWN_ZONES = auto()
    SPAWNS = auto()
    GOLD_MINES = auto()
    CREEP_CAMPS = auto()
    DEBUG_DISTANCES = auto()
    DEBUG_TEAM_BOUNDARIES = auto()


# =============================================================================
# Colors
# =============================================================================
COLOR_WORLD_EDGE = (150, 0, 0)
COLOR_LAND = (34, 139, 34)           # Forest green
COLOR_WATER = (30, 60, 120)          # Deep blue
COLOR_SPAWN = (255, 215, 0)          # Gold for player spawns
COLOR_GOLD_MINE = (255, 200, 0)      # Yellow-gold
COLOR_GOLD_MINE_NEUTRAL = (218, 165, 32)  # Goldenrod for neutral mines

# Terrain feature colors
COLOR_FOREST = (20, 90, 20)          # Dark forest green
COLOR_DENSE_FOREST = (10, 50, 10)    # Very dark forest
COLOR_TRAIL = (139, 119, 101)        # Light brown trail
COLOR_RIVER = (65, 105, 225)         # Royal blue water
COLOR_BRIDGE = (139, 90, 43)         # Brown wood
COLOR_CLIFF = (105, 105, 105)        # Gray rock

# Creep camp colors by difficulty
COLOR_CREEP_EASY = (144, 238, 144)   # Light green
COLOR_CREEP_MEDIUM = (255, 165, 0)   # Orange
COLOR_CREEP_HARD = (220, 20, 60)     # Crimson

# Team colors (used for spawn area highlighting)
TEAM_COLORS = [
    (65, 105, 225),   # Royal Blue
    (220, 20, 60),    # Crimson
    (50, 205, 50),    # Lime Green
    (255, 140, 0),    # Dark Orange
    (138, 43, 226),   # Blue Violet
    (0, 206, 209),    # Dark Turquoise
    (255, 20, 147),   # Deep Pink
    (255, 255, 0),    # Yellow
]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class MyWorldConfig:
    """Configuration for WC3-style strategy game map generation."""
    width: int = 2048
    height: int = 1536
    tile_size: float = 16.0
    seed: int = 42

    # Team configuration: list of team sizes, e.g. [2, 2] = 2 teams of 2 players
    teams: list[int] = field(default_factory=lambda: [2, 2])

    # Spawn placement
    spawn_radius_ratio: float = 0.35   # How far from center (0.0-0.5)
    teammate_arc_degrees: float = 30   # Angle spread for teammates

    # Gold mines
    gold_mines_per_player: int = 1     # Starting mines per player
    neutral_gold_mines: int = 4        # Additional contested mines
    gold_mine_offset_tiles: int = 3    # Distance from spawn to starting mine
    neutral_mine_min_dist: float = 200 # Min distance from any spawn for neutral mines

    # Creep camps
    creep_density: float = 1.0         # Multiplier for map size calculation

    # Terrain
    land_ratio: float = 0.85           # How much of the map should be land
    spawn_flat_radius: int = 8         # Tiles around spawn guaranteed flat

    # Phase 5: Trails
    trail_cost_reduction: float = 0.7  # Movement cost on trails
    trail_width: int = 2               # Width of trail in tiles

    # Phase 6: Forests
    forest_coverage: float = 0.35      # Target 35% coverage
    forest_ca_iterations: int = 5      # Cellular automata iterations
    forest_initial_density: float = 0.45  # Initial random density
    forest_clear_radius_spawn: int = 10   # Clear radius around spawns
    forest_clear_radius_objective: int = 3  # Clear radius around objectives

    # Phase 7: Separation
    river_width: int = 3               # Width of rivers in tiles
    bridges_per_boundary: int = 2      # Bridge crossings per team boundary
    cliff_threshold: float = 0.3       # Elevation difference for cliffs
    dense_forest_width: int = 5        # Width of dense forest barriers
    separation_distance: float = 0.15  # Distance from center for separators (ratio)


def colormap(t: float, base_color: Color) -> Color:
    """Apply brightness mapping to a base color."""
    r, g, b = base_color
    t = clamp(t, 0.3, 1.0)  # Don't go too dark
    return int(r * t), int(g * t), int(b * t)


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class PlayerSpawn:
    """Represents a player's spawn point."""
    player_id: int
    team_id: int
    position: Vec2       # World coordinates
    tile: GVec2          # Tile coordinates
    angle: float         # Angle from center (radians)


@dataclass
class GoldMine:
    """Represents a gold mine location."""
    position: Vec2
    tile: GVec2
    is_starting: bool    # True if it's a player's starting mine
    owner_player: int | None  # Player ID if starting mine, None if neutral


@dataclass
class CreepCamp:
    """Represents a neutral creep camp."""
    position: Vec2
    tile: GVec2
    difficulty: CreepDifficulty
    distance_to_nearest_spawn: float


@dataclass
class Trail:
    """Represents a trail path between points."""
    tiles: list[GVec2]
    from_player: int
    to_player: int
    is_teammate_path: bool  # True if between teammates


@dataclass
class River:
    """Represents a river barrier between teams."""
    tiles: set[GVec2]
    bridge_tiles: set[GVec2]
    separates_teams: tuple[int, int]


@dataclass
class Cliff:
    """Represents a cliff barrier."""
    tiles: set[GVec2]
    pass_tiles: set[GVec2]  # Passable gaps


# =============================================================================
# Main World Class
# =============================================================================
class MyWorld:
    """WC3-style strategy game world generator."""

    def __init__(self, cfg: MyWorldConfig = MyWorldConfig()):
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        self.ts = cfg.tile_size

        # Generated data
        self.spawns: list[PlayerSpawn] = []
        self.gold_mines: list[GoldMine] = []
        self.creep_camps: list[CreepCamp] = []
        self.trails: list[Trail] = []
        self.rivers: list[River] = []
        self.cliffs: list[Cliff] = []

        # Terrain arrays
        self.elevation: np.ndarray | None = None
        self.is_land: np.ndarray | None = None
        self.terrain_type: np.ndarray | None = None
        self.is_trail: np.ndarray | None = None
        self.is_forest: np.ndarray | None = None

        self._regenerate()

        # Visualization layers
        self.layers = {
            ViewLayers.BORDER: DrawLayer(z=1, label="world_border", draw=self._draw_world_border),
            ViewLayers.TERRAIN: DrawLayer(z=10, label="terrain", draw=self._draw_terrain),
            ViewLayers.FORESTS: DrawLayer(z=12, label="forests", draw=self._draw_forests),
            ViewLayers.RIVERS: DrawLayer(z=14, label="rivers", draw=self._draw_rivers),
            ViewLayers.CLIFFS: DrawLayer(z=16, label="cliffs", draw=self._draw_cliffs),
            ViewLayers.TRAILS: DrawLayer(z=18, label="trails", draw=self._draw_trails),
            ViewLayers.SPAWN_ZONES: DrawLayer(z=20, label="spawn_zones", draw=self._draw_spawn_zones),
            ViewLayers.SPAWNS: DrawLayer(z=22, label="spawns", draw=self._draw_spawns),
            ViewLayers.GOLD_MINES: DrawLayer(z=24, label="gold_mines", draw=self._draw_gold_mines),
            ViewLayers.CREEP_CAMPS: DrawLayer(z=26, label="creep_camps", draw=self._draw_creep_camps),
            ViewLayers.DEBUG_DISTANCES: DrawLayer(z=50, label="debug_distances", draw=self._draw_debug_distances),
            ViewLayers.DEBUG_TEAM_BOUNDARIES: DrawLayer(z=52, label="debug_boundaries", draw=self._draw_team_boundaries),
        }

    def get_layers(self, which: Iterable[ViewLayers]) -> list[DrawLayer]:
        """Get specific visualization layers."""
        return [self.layers[layer] for layer in which]

    def debug_layers(self) -> list[DrawLayer]:
        """Get debug visualization layers."""
        return [
            DrawLayer(z=40, label="elevation", draw=lambda ctx: self._draw_array(ctx, self.elevation)),
            DrawLayer(z=42, label="terrain_type", draw=lambda ctx: self._draw_array(ctx, self.terrain_type.astype(float))),
        ]

    # =========================================================================
    # Coordinate Utilities
    # =========================================================================
    @property
    def shape_tiles(self) -> tuple[int, int]:
        """Map dimensions in tiles."""
        return int(math.floor(self.width / self.ts)), int(math.floor(self.height / self.ts))

    @property
    def center(self) -> Vec2:
        """Map center in world coordinates."""
        return self.width / 2, self.height / 2

    @property
    def spawn_radius(self) -> float:
        """Radius from center for player spawns."""
        return min(self.width, self.height) * self.cfg.spawn_radius_ratio

    def _tile_at_world(self, p: Vec2) -> GVec2:
        """Convert world coordinates to tile coordinates."""
        return int(math.floor(p[0] / self.ts)), int(math.floor(p[1] / self.ts))

    def _tile_to_world(self, tile: GVec2) -> Vec2:
        """Convert tile coordinates to world coordinates (tile center)."""
        return (tile[0] + 0.5) * self.ts, (tile[1] + 0.5) * self.ts

    def _tile_to_world_corner(self, tile: GVec2) -> Vec2:
        """Convert tile coordinates to world coordinates (tile corner)."""
        return tile[0] * self.ts, tile[1] * self.ts

    def is_tile_in_bounds(self, tile: GVec2) -> bool:
        """Check if a tile is within map bounds."""
        tw, th = self.shape_tiles
        return 0 <= tile[0] < tw and 0 <= tile[1] < th

    def _distance(self, p1: Vec2, p2: Vec2) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _tile_distance(self, t1: GVec2, t2: GVec2) -> float:
        """Euclidean distance between two tiles."""
        return math.sqrt((t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2)

    # =========================================================================
    # Generation Pipeline
    # =========================================================================
    def _regenerate(self):
        """Run the complete generation pipeline."""
        self._generate_spawns()
        self._generate_gold_mines()
        self._generate_creep_camps()
        self._generate_terrain()           # Phase 4: Base terrain
        self._generate_trails()            # Phase 5: Trails between teammates
        self._generate_team_separators()   # Phase 7: Rivers, cliffs, dense forests
        self._generate_forests()           # Phase 6: Forest coverage (respects trails/separators)

    # =========================================================================
    # Phase 1: Radial Spawn Point Generation
    # =========================================================================
    def _generate_spawns(self):
        """Generate player spawn points using radial placement."""
        rng = random.Random(self.cfg.seed)
        self.spawns = []

        teams = self.cfg.teams
        num_teams = len(teams)
        total_players = sum(teams)

        if num_teams == 0 or total_players == 0:
            return

        cx, cy = self.center
        radius = self.spawn_radius

        # Base angle for each team (evenly distributed around circle)
        team_base_angles = [2 * math.pi * i / num_teams for i in range(num_teams)]

        # Add some random rotation so maps aren't always aligned the same way
        rotation_offset = rng.uniform(0, 2 * math.pi)
        team_base_angles = [(a + rotation_offset) % (2 * math.pi) for a in team_base_angles]

        player_id = 0
        for team_id, team_size in enumerate(teams):
            base_angle = team_base_angles[team_id]

            # Calculate angle spread for teammates
            if team_size == 1:
                angles = [base_angle]
            else:
                arc_rad = math.radians(self.cfg.teammate_arc_degrees)
                start_angle = base_angle - arc_rad / 2
                angle_step = arc_rad / (team_size - 1) if team_size > 1 else 0
                angles = [start_angle + i * angle_step for i in range(team_size)]

            for angle in angles:
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)

                tile = self._tile_at_world((x, y))
                position = self._tile_to_world(tile)

                spawn = PlayerSpawn(
                    player_id=player_id,
                    team_id=team_id,
                    position=position,
                    tile=tile,
                    angle=angle
                )
                self.spawns.append(spawn)
                player_id += 1

    # =========================================================================
    # Phase 2: Gold Mine Placement
    # =========================================================================
    def _generate_gold_mines(self):
        """Generate gold mines - starting mines and neutral contested mines."""
        rng = random.Random(self.cfg.seed + 100)
        self.gold_mines = []

        cx, cy = self.center

        # 1. Starting mines for each player
        for spawn in self.spawns:
            for _ in range(self.cfg.gold_mines_per_player):
                dx = cx - spawn.position[0]
                dy = cy - spawn.position[1]
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > 0:
                    offset = self.cfg.gold_mine_offset_tiles * self.ts
                    mx = spawn.position[0] + (dx / dist) * offset
                    my = spawn.position[1] + (dy / dist) * offset
                else:
                    mx, my = spawn.position

                tile = self._tile_at_world((mx, my))
                position = self._tile_to_world(tile)

                mine = GoldMine(
                    position=position,
                    tile=tile,
                    is_starting=True,
                    owner_player=spawn.player_id
                )
                self.gold_mines.append(mine)

        # 2. Neutral mines in contested areas
        spawn_positions = [s.position for s in self.spawns]
        mine_positions = [m.position for m in self.gold_mines]

        neutral_placed = 0
        max_attempts = self.cfg.neutral_gold_mines * 100

        for _ in range(max_attempts):
            if neutral_placed >= self.cfg.neutral_gold_mines:
                break

            x = rng.gauss(cx, self.width * 0.2)
            y = rng.gauss(cy, self.height * 0.2)

            margin = self.ts * 5
            x = clamp(x, margin, self.width - margin)
            y = clamp(y, margin, self.height - margin)

            too_close_to_spawn = any(
                self._distance((x, y), sp) < self.cfg.neutral_mine_min_dist
                for sp in spawn_positions
            )
            if too_close_to_spawn:
                continue

            min_mine_dist = self.ts * 8
            too_close_to_mine = any(
                self._distance((x, y), mp) < min_mine_dist
                for mp in mine_positions
            )
            if too_close_to_mine:
                continue

            tile = self._tile_at_world((x, y))
            position = self._tile_to_world(tile)

            mine = GoldMine(
                position=position,
                tile=tile,
                is_starting=False,
                owner_player=None
            )
            self.gold_mines.append(mine)
            mine_positions.append(position)
            neutral_placed += 1

    # =========================================================================
    # Phase 3: Neutral Creep Camp Placement
    # =========================================================================
    def _generate_creep_camps(self):
        """Generate neutral creep camps with distance-based difficulty tiers."""
        rng = random.Random(self.cfg.seed + 200)
        self.creep_camps = []

        map_area_tiles = (self.width * self.height) / (self.ts ** 2)
        base_camps = int(map_area_tiles / 1500 * self.cfg.creep_density)

        target_easy = int(base_camps * 0.50)
        target_medium = int(base_camps * 0.35)
        target_hard = int(base_camps * 0.15)

        counts = {d: 0 for d in CreepDifficulty}
        targets = {
            CreepDifficulty.EASY: target_easy,
            CreepDifficulty.MEDIUM: target_medium,
            CreepDifficulty.HARD: target_hard,
        }

        candidates = poisson_disc_2d(
            bounds=(self.ts * 3, self.ts * 3, self.width - self.ts * 3, self.height - self.ts * 3),
            radius=self.ts * 6,
            n_points=base_camps * 3,
            seed=self.cfg.seed + 201
        )

        spawn_positions = [s.position for s in self.spawns]
        mine_positions = [m.position for m in self.gold_mines]

        easy_max_dist = self.spawn_radius * 0.4
        medium_max_dist = self.spawn_radius * 0.8

        for candidate in candidates:
            if all(counts[d] >= targets[d] for d in CreepDifficulty):
                break

            x, y = candidate

            min_spawn_dist = self.ts * 5
            if any(self._distance((x, y), sp) < min_spawn_dist for sp in spawn_positions):
                continue

            min_mine_dist = self.ts * 4
            if any(self._distance((x, y), mp) < min_mine_dist for mp in mine_positions):
                continue

            dist_to_nearest = min(self._distance((x, y), sp) for sp in spawn_positions)

            if dist_to_nearest < easy_max_dist:
                difficulty = CreepDifficulty.EASY
            elif dist_to_nearest < medium_max_dist:
                difficulty = CreepDifficulty.MEDIUM
            else:
                difficulty = CreepDifficulty.HARD

            if counts[difficulty] >= targets[difficulty]:
                continue

            tile = self._tile_at_world((x, y))
            position = self._tile_to_world(tile)

            camp = CreepCamp(
                position=position,
                tile=tile,
                difficulty=difficulty,
                distance_to_nearest_spawn=dist_to_nearest
            )
            self.creep_camps.append(camp)
            counts[difficulty] += 1

    # =========================================================================
    # Phase 4: Terrain Generation
    # =========================================================================
    def _generate_terrain(self):
        """Generate terrain - mostly land for strategy games."""
        shape = self.shape_tiles

        self.elevation = np.empty(shape, dtype=float)
        self.is_land = np.ones(shape, dtype=bool)
        self.terrain_type = np.full(shape, TerrainType.LAND, dtype=int)
        self.is_trail = np.zeros(shape, dtype=bool)
        self.is_forest = np.zeros(shape, dtype=bool)

        cx, cy = self.center
        max_dist = math.sqrt(cx ** 2 + cy ** 2)

        # Generate elevation using Perlin noise
        for tile in np.ndindex(shape):
            x, y = self._tile_to_world(tile)

            n1 = 0.5 * (perlin2d_fbm(x / 256, y / 256, octaves=3, seed=self.cfg.seed) + 1)
            n2 = 0.5 * (perlin2d_fbm(x / 64, y / 64, octaves=2, seed=self.cfg.seed + 1) + 1)

            base_elevation = 0.6 * n1 + 0.4 * n2

            dist_from_center = self._distance((x, y), (cx, cy))
            edge_factor = dist_from_center / max_dist

            edge_reduction = max(0, (edge_factor - 0.6) * 2.5)
            elevation = base_elevation - edge_reduction

            self.elevation[tile] = clamp(elevation, 0, 1)

            water_threshold = 1.0 - self.cfg.land_ratio
            if self.elevation[tile] <= water_threshold * 0.5:
                self.is_land[tile] = False
                self.terrain_type[tile] = TerrainType.WATER

        # Ensure spawn areas are flat land
        spawn_flat_radius = self.cfg.spawn_flat_radius
        for spawn in self.spawns:
            sx, sy = spawn.tile
            for dx in range(-spawn_flat_radius, spawn_flat_radius + 1):
                for dy in range(-spawn_flat_radius, spawn_flat_radius + 1):
                    tx, ty = sx + dx, sy + dy
                    if self.is_tile_in_bounds((tx, ty)):
                        self.is_land[tx, ty] = True
                        self.terrain_type[tx, ty] = TerrainType.LAND
                        dist = math.sqrt(dx ** 2 + dy ** 2)
                        if dist <= spawn_flat_radius:
                            blend = dist / spawn_flat_radius
                            self.elevation[tx, ty] = max(
                                0.5,
                                self.elevation[tx, ty] * blend + 0.6 * (1 - blend)
                            )

        # Ensure gold mine and creep camp areas are on land
        for mine in self.gold_mines:
            mx, my = mine.tile
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    tx, ty = mx + dx, my + dy
                    if self.is_tile_in_bounds((tx, ty)):
                        self.is_land[tx, ty] = True
                        if self.terrain_type[tx, ty] == TerrainType.WATER:
                            self.terrain_type[tx, ty] = TerrainType.LAND

        for camp in self.creep_camps:
            ccx, ccy = camp.tile
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    tx, ty = ccx + dx, ccy + dy
                    if self.is_tile_in_bounds((tx, ty)):
                        self.is_land[tx, ty] = True
                        if self.terrain_type[tx, ty] == TerrainType.WATER:
                            self.terrain_type[tx, ty] = TerrainType.LAND

    # =========================================================================
    # Phase 5: Trail Generation (Dijkstra-based)
    # =========================================================================
    def _generate_trails(self):
        """Generate natural trails between teammates using Dijkstra pathfinding."""
        self.trails = []
        shape = self.shape_tiles

        if len(self.spawns) < 2:
            return

        # Group spawns by team
        teams_spawns: dict[int, list[PlayerSpawn]] = defaultdict(list)
        for spawn in self.spawns:
            teams_spawns[spawn.team_id].append(spawn)

        # Cost function for pathfinding
        def passable(x: int, y: int) -> bool:
            if not self.is_tile_in_bounds((x, y)):
                return False
            return self.is_land[x, y]

        def cost_of(x: int, y: int, nx: int, ny: int) -> float:
            if not self.is_tile_in_bounds((nx, ny)):
                return float('inf')
            base_cost = 1.0
            # Higher cost for elevated terrain
            elev = self.elevation[nx, ny]
            if elev > 0.7:
                base_cost = 2.5
            elif elev > 0.5:
                base_cost = 1.5
            return base_cost

        # Create trails between teammates
        for team_id, team_spawns in teams_spawns.items():
            if len(team_spawns) < 2:
                continue

            # Connect all teammates (create a path network)
            for i, spawn1 in enumerate(team_spawns):
                for spawn2 in team_spawns[i + 1:]:
                    # Find path using Dijkstra
                    _, parent = dijkstra_grid(
                        width=shape[0],
                        height=shape[1],
                        passable=passable,
                        cost_of=cost_of,
                        start=spawn1.tile,
                        goal=spawn2.tile
                    )

                    path = reconstruct_path_grid(parent, spawn1.tile, spawn2.tile)

                    if path and len(path) > 1:
                        trail = Trail(
                            tiles=path,
                            from_player=spawn1.player_id,
                            to_player=spawn2.player_id,
                            is_teammate_path=True
                        )
                        self.trails.append(trail)

                        # Mark trail tiles with width
                        for tile in path:
                            self._mark_trail_area(tile)

    def _mark_trail_area(self, center: GVec2):
        """Mark tiles around a trail center as trail tiles."""
        width = self.cfg.trail_width
        cx, cy = center
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                tx, ty = cx + dx, cy + dy
                if self.is_tile_in_bounds((tx, ty)) and self.is_land[tx, ty]:
                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    if dist <= width:
                        self.is_trail[tx, ty] = True

    # =========================================================================
    # Phase 7: Team Separation Features (Rivers, Cliffs, Dense Forests)
    # =========================================================================
    def _generate_team_separators(self):
        """Generate barriers between opposing teams."""
        self.rivers = []
        self.cliffs = []

        num_teams = len(self.cfg.teams)
        if num_teams < 2:
            return

        rng = random.Random(self.cfg.seed + 400)

        # Get team center angles
        team_angles = {}
        for spawn in self.spawns:
            if spawn.team_id not in team_angles:
                team_angles[spawn.team_id] = spawn.angle

        # Calculate boundary angles (midpoints between teams)
        sorted_teams = sorted(team_angles.items(), key=lambda x: x[1])
        boundary_angles = []

        for i in range(len(sorted_teams)):
            team1_id, angle1 = sorted_teams[i]
            team2_id, angle2 = sorted_teams[(i + 1) % len(sorted_teams)]

            # Calculate midpoint angle
            if angle2 < angle1:
                angle2 += 2 * math.pi

            mid_angle = (angle1 + angle2) / 2
            if mid_angle > 2 * math.pi:
                mid_angle -= 2 * math.pi

            boundary_angles.append((mid_angle, team1_id, team2_id))

        # Generate separation features along each boundary
        cx, cy = self.center
        sep_start = min(self.width, self.height) * self.cfg.separation_distance
        sep_end = min(self.width, self.height) * 0.45

        for boundary_angle, team1, team2 in boundary_angles:
            # Decide which type of separator to use based on terrain
            separator_type = rng.choice(['river', 'cliff', 'dense_forest'])

            if separator_type == 'river':
                self._generate_river(boundary_angle, sep_start, sep_end, team1, team2, rng)
            elif separator_type == 'cliff':
                self._generate_cliff(boundary_angle, sep_start, sep_end, rng)
            else:
                self._generate_dense_forest_barrier(boundary_angle, sep_start, sep_end)

    def _generate_river(self, angle: float, start_dist: float, end_dist: float,
                        team1: int, team2: int, rng: random.Random):
        """Generate a meandering river along a boundary angle."""
        cx, cy = self.center
        river_tiles: set[GVec2] = set()
        bridge_tiles: set[GVec2] = set()

        # Generate river path with Perlin noise for meandering
        num_points = 50
        bridge_positions = set(rng.sample(range(num_points), min(self.cfg.bridges_per_boundary, num_points)))

        for i in range(num_points):
            t = i / (num_points - 1)
            dist = start_dist + t * (end_dist - start_dist)

            # Add noise to angle for meandering
            noise_val = perlin2d_fbm(t * 5, angle * 10, octaves=2, seed=self.cfg.seed + 500)
            angle_offset = noise_val * 0.2  # Max ~11 degree deviation

            actual_angle = angle + angle_offset

            x = cx + dist * math.cos(actual_angle)
            y = cy + dist * math.sin(actual_angle)

            tile = self._tile_at_world((x, y))

            # Add width to river
            for dx in range(-self.cfg.river_width // 2, self.cfg.river_width // 2 + 1):
                for dy in range(-self.cfg.river_width // 2, self.cfg.river_width // 2 + 1):
                    tx, ty = tile[0] + dx, tile[1] + dy
                    if self.is_tile_in_bounds((tx, ty)):
                        if i in bridge_positions and abs(dx) <= 1 and abs(dy) <= 1:
                            bridge_tiles.add((tx, ty))
                            self.terrain_type[tx, ty] = TerrainType.BRIDGE
                            self.is_land[tx, ty] = True
                        else:
                            river_tiles.add((tx, ty))
                            self.terrain_type[tx, ty] = TerrainType.RIVER
                            self.is_land[tx, ty] = False

        river = River(
            tiles=river_tiles,
            bridge_tiles=bridge_tiles,
            separates_teams=(team1, team2)
        )
        self.rivers.append(river)

    def _generate_cliff(self, angle: float, start_dist: float, end_dist: float,
                        rng: random.Random):
        """Generate cliffs along a boundary where elevation varies."""
        cx, cy = self.center
        cliff_tiles: set[GVec2] = set()
        pass_tiles: set[GVec2] = set()

        num_points = 40
        pass_positions = set(rng.sample(range(num_points), min(2, num_points)))

        for i in range(num_points):
            t = i / (num_points - 1)
            dist = start_dist + t * (end_dist - start_dist)

            # Add noise for irregular edge
            noise_val = perlin2d_fbm(t * 4, angle * 8, octaves=2, seed=self.cfg.seed + 600)
            angle_offset = noise_val * 0.15

            actual_angle = angle + angle_offset

            x = cx + dist * math.cos(actual_angle)
            y = cy + dist * math.sin(actual_angle)

            tile = self._tile_at_world((x, y))

            # Add width to cliff
            width = 2
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    tx, ty = tile[0] + dx, tile[1] + dy
                    if self.is_tile_in_bounds((tx, ty)):
                        if i in pass_positions:
                            pass_tiles.add((tx, ty))
                            # Keep as passable land
                        else:
                            cliff_tiles.add((tx, ty))
                            self.terrain_type[tx, ty] = TerrainType.CLIFF
                            # Cliffs are impassable in most games
                            self.is_land[tx, ty] = False

        cliff = Cliff(tiles=cliff_tiles, pass_tiles=pass_tiles)
        self.cliffs.append(cliff)

    def _generate_dense_forest_barrier(self, angle: float, start_dist: float, end_dist: float):
        """Generate a dense forest barrier along a boundary."""
        cx, cy = self.center

        num_points = 50
        width = self.cfg.dense_forest_width

        for i in range(num_points):
            t = i / (num_points - 1)
            dist = start_dist + t * (end_dist - start_dist)

            # Add noise for irregular edge
            noise_val = perlin2d_fbm(t * 3, angle * 6, octaves=2, seed=self.cfg.seed + 700)
            angle_offset = noise_val * 0.1

            actual_angle = angle + angle_offset

            x = cx + dist * math.cos(actual_angle)
            y = cy + dist * math.sin(actual_angle)

            tile = self._tile_at_world((x, y))

            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    tx, ty = tile[0] + dx, tile[1] + dy
                    if self.is_tile_in_bounds((tx, ty)) and self.is_land[tx, ty]:
                        # Don't overwrite trails or other features
                        if self.terrain_type[tx, ty] == TerrainType.LAND and not self.is_trail[tx, ty]:
                            self.terrain_type[tx, ty] = TerrainType.DENSE_FOREST
                            self.is_forest[tx, ty] = True

    # =========================================================================
    # Phase 6: Forest Generation (Cellular Automata)
    # =========================================================================
    def _generate_forests(self):
        """Generate forests using cellular automata."""
        rng = random.Random(self.cfg.seed + 300)
        shape = self.shape_tiles

        # Initialize forest map with random seeds
        forest_map = np.zeros(shape, dtype=bool)

        for tile in np.ndindex(shape):
            if self.is_land[tile] and self.terrain_type[tile] == TerrainType.LAND:
                if rng.random() < self.cfg.forest_initial_density:
                    forest_map[tile] = True

        # Run cellular automata iterations
        for _ in range(self.cfg.forest_ca_iterations):
            new_forest = np.zeros(shape, dtype=bool)

            for x in range(shape[0]):
                for y in range(shape[1]):
                    if not self.is_land[x, y] or self.terrain_type[x, y] != TerrainType.LAND:
                        continue

                    # Count forest neighbors (8-connected)
                    neighbors = 0
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if self.is_tile_in_bounds((nx, ny)) and forest_map[nx, ny]:
                                neighbors += 1

                    # CA rules: 5+ neighbors -> forest, <4 neighbors -> clear
                    if forest_map[x, y]:
                        new_forest[x, y] = neighbors >= 4
                    else:
                        new_forest[x, y] = neighbors >= 5

            forest_map = new_forest

        # Post-process: Clear areas around spawns
        for spawn in self.spawns:
            sx, sy = spawn.tile
            clear_radius = self.cfg.forest_clear_radius_spawn
            for dx in range(-clear_radius, clear_radius + 1):
                for dy in range(-clear_radius, clear_radius + 1):
                    tx, ty = sx + dx, sy + dy
                    if self.is_tile_in_bounds((tx, ty)):
                        dist = math.sqrt(dx ** 2 + dy ** 2)
                        if dist <= clear_radius:
                            forest_map[tx, ty] = False

        # Clear areas around objectives (gold mines, creep camps)
        clear_radius = self.cfg.forest_clear_radius_objective
        for mine in self.gold_mines:
            mx, my = mine.tile
            for dx in range(-clear_radius, clear_radius + 1):
                for dy in range(-clear_radius, clear_radius + 1):
                    tx, ty = mx + dx, my + dy
                    if self.is_tile_in_bounds((tx, ty)):
                        forest_map[tx, ty] = False

        for camp in self.creep_camps:
            ccx, ccy = camp.tile
            for dx in range(-clear_radius, clear_radius + 1):
                for dy in range(-clear_radius, clear_radius + 1):
                    tx, ty = ccx + dx, ccy + dy
                    if self.is_tile_in_bounds((tx, ty)):
                        forest_map[tx, ty] = False

        # Reduce forest on trails (50% chance to clear)
        for tile in np.ndindex(shape):
            if self.is_trail[tile] and forest_map[tile]:
                if rng.random() < 0.5:
                    forest_map[tile] = False

        # Apply forest to terrain type
        for tile in np.ndindex(shape):
            if forest_map[tile] and self.terrain_type[tile] == TerrainType.LAND:
                self.terrain_type[tile] = TerrainType.FOREST
                self.is_forest[tile] = True

    # =========================================================================
    # Drawing Methods
    # =========================================================================
    def _draw_world_border(self, ctx: RenderContext) -> None:
        """Draw the world boundary."""
        w, h = self.width, self.height
        origin = ctx.camera.world_to_screen((0, 0))
        corner = ctx.camera.world_to_screen((w, h))
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, origin, (corner[0], origin[1]), 3)
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, origin, (origin[0], corner[1]), 3)
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, corner, (origin[0], corner[1]), 3)
        pygame.draw.line(ctx.screen, COLOR_WORLD_EDGE, corner, (corner[0], origin[1]), 3)

    def _draw_terrain(self, ctx: RenderContext) -> None:
        """Draw the base terrain (land/water with elevation shading)."""
        ts = self.ts
        for tile in np.ndindex(self.shape_tiles):
            terrain = self.terrain_type[tile]
            elevation = self.elevation[tile]

            if terrain == TerrainType.WATER:
                color = colormap(elevation + 0.3, COLOR_WATER)
            elif terrain == TerrainType.RIVER:
                color = colormap(0.8, COLOR_RIVER)
            elif terrain == TerrainType.BRIDGE:
                color = COLOR_BRIDGE
            elif terrain == TerrainType.CLIFF:
                color = colormap(elevation, COLOR_CLIFF)
            elif terrain == TerrainType.DENSE_FOREST:
                color = colormap(elevation, COLOR_DENSE_FOREST)
            elif terrain == TerrainType.FOREST:
                color = colormap(elevation, COLOR_FOREST)
            else:
                color = colormap(elevation, COLOR_LAND)

            draw_tile(ctx, tile, ts, color)

    def _draw_forests(self, ctx: RenderContext) -> None:
        """Draw forest overlay with tree markers."""
        ts = self.ts
        for tile in np.ndindex(self.shape_tiles):
            if self.is_forest[tile]:
                # Draw small tree symbol
                center = ctx.camera.world_to_screen(self._tile_to_world(tile))
                size = int(ts * ctx.camera.zoom * 0.25)
                if size >= 2:
                    # Simple tree shape (triangle)
                    points = [
                        (center[0], center[1] - size),
                        (center[0] - size, center[1] + size),
                        (center[0] + size, center[1] + size),
                    ]
                    color = (0, 60, 0) if self.terrain_type[tile] == TerrainType.DENSE_FOREST else (0, 80, 0)
                    pygame.draw.polygon(ctx.screen, color, points)

    def _draw_rivers(self, ctx: RenderContext) -> None:
        """Draw river highlights and bridge markers."""
        ts = self.ts
        for river in self.rivers:
            # Draw bridge markers
            for bridge_tile in river.bridge_tiles:
                center = ctx.camera.world_to_screen(self._tile_to_world(bridge_tile))
                size = int(ts * ctx.camera.zoom * 0.4)
                pygame.draw.rect(ctx.screen, (139, 69, 19),
                               (center[0] - size, center[1] - size // 2, size * 2, size), 2)

    def _draw_cliffs(self, ctx: RenderContext) -> None:
        """Draw cliff edge highlights."""
        ts = self.ts
        for cliff in self.cliffs:
            for cliff_tile in cliff.tiles:
                if self.is_tile_in_bounds(cliff_tile):
                    center = ctx.camera.world_to_screen(self._tile_to_world(cliff_tile))
                    size = int(ts * ctx.camera.zoom * 0.3)
                    # Draw rocky texture indicator
                    pygame.draw.circle(ctx.screen, (80, 80, 80), center, size, 1)

    def _draw_trails(self, ctx: RenderContext) -> None:
        """Draw trail paths."""
        ts = self.ts
        for tile in np.ndindex(self.shape_tiles):
            if self.is_trail[tile] and self.terrain_type[tile] not in (TerrainType.WATER, TerrainType.RIVER):
                # Draw trail overlay
                pos = ctx.camera.world_to_screen(self._tile_to_world_corner(tile))
                size = int(ts * ctx.camera.zoom)
                # Semi-transparent trail color
                trail_surface = pygame.Surface((size, size), pygame.SRCALPHA)
                trail_surface.fill((*COLOR_TRAIL, 100))
                ctx.screen.blit(trail_surface, pos)

    def _draw_spawn_zones(self, ctx: RenderContext) -> None:
        """Draw highlighted zones around spawns showing team territories."""
        ts = self.ts
        spawn_radius = self.cfg.spawn_flat_radius

        for spawn in self.spawns:
            team_color = TEAM_COLORS[spawn.team_id % len(TEAM_COLORS)]
            sx, sy = spawn.tile

            for dx in range(-spawn_radius, spawn_radius + 1):
                for dy in range(-spawn_radius, spawn_radius + 1):
                    tx, ty = sx + dx, sy + dy
                    if not self.is_tile_in_bounds((tx, ty)):
                        continue

                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    if dist <= spawn_radius:
                        alpha = 0.3 * (1 - dist / spawn_radius)
                        terrain = self.terrain_type[tx, ty]
                        if terrain == TerrainType.WATER:
                            base = COLOR_WATER
                        elif terrain == TerrainType.FOREST:
                            base = COLOR_FOREST
                        else:
                            base = COLOR_LAND

                        blended = tuple(
                            int(base[i] * (1 - alpha) + team_color[i] * alpha)
                            for i in range(3)
                        )
                        draw_tile(ctx, (tx, ty), ts, blended)

    def _draw_spawns(self, ctx: RenderContext) -> None:
        """Draw player spawn points."""
        ts = self.ts
        for spawn in self.spawns:
            team_color = TEAM_COLORS[spawn.team_id % len(TEAM_COLORS)]
            draw_tile(ctx, spawn.tile, ts, team_color)

            pos = ctx.camera.world_to_screen(self._tile_to_world_corner(spawn.tile))
            size = int(ts * ctx.camera.zoom)
            pygame.draw.rect(ctx.screen, COLOR_SPAWN, (pos[0], pos[1], size, size), 2)

    def _draw_gold_mines(self, ctx: RenderContext) -> None:
        """Draw gold mine locations."""
        ts = self.ts
        for mine in self.gold_mines:
            color = COLOR_GOLD_MINE if mine.is_starting else COLOR_GOLD_MINE_NEUTRAL
            draw_tile(ctx, mine.tile, ts, color)

            center = ctx.camera.world_to_screen(mine.position)
            size = int(ts * ctx.camera.zoom * 0.4)
            points = [
                (center[0], center[1] - size),
                (center[0] + size, center[1]),
                (center[0], center[1] + size),
                (center[0] - size, center[1]),
            ]
            pygame.draw.polygon(ctx.screen, (255, 255, 255), points, 2)

    def _draw_creep_camps(self, ctx: RenderContext) -> None:
        """Draw neutral creep camps with difficulty-based colors."""
        ts = self.ts
        difficulty_colors = {
            CreepDifficulty.EASY: COLOR_CREEP_EASY,
            CreepDifficulty.MEDIUM: COLOR_CREEP_MEDIUM,
            CreepDifficulty.HARD: COLOR_CREEP_HARD,
        }

        for camp in self.creep_camps:
            color = difficulty_colors[camp.difficulty]
            draw_tile(ctx, camp.tile, ts, color)

            center = ctx.camera.world_to_screen(camp.position)
            size = int(ts * ctx.camera.zoom * 0.3)
            pygame.draw.circle(ctx.screen, (50, 50, 50), center, size)
            pygame.draw.circle(ctx.screen, color, center, size, 2)

    def _draw_debug_distances(self, ctx: RenderContext) -> None:
        """Debug visualization showing distance from spawns."""
        if not self.spawns:
            return

        ts = self.ts
        spawn_positions = [s.position for s in self.spawns]
        max_dist = self.spawn_radius * 1.5

        for tile in np.ndindex(self.shape_tiles):
            pos = self._tile_to_world(tile)
            min_dist = min(self._distance(pos, sp) for sp in spawn_positions)

            t = clamp(min_dist / max_dist, 0, 1)
            color = (int(255 * t), int(255 * (1 - t)), 0)
            draw_tile(ctx, tile, ts, color)

    def _draw_team_boundaries(self, ctx: RenderContext) -> None:
        """Debug visualization showing team boundary lines."""
        if len(self.cfg.teams) < 2:
            return

        cx, cy = self.center

        # Get team angles
        team_angles = {}
        for spawn in self.spawns:
            if spawn.team_id not in team_angles:
                team_angles[spawn.team_id] = spawn.angle

        sorted_teams = sorted(team_angles.items(), key=lambda x: x[1])

        for i in range(len(sorted_teams)):
            _, angle1 = sorted_teams[i]
            _, angle2 = sorted_teams[(i + 1) % len(sorted_teams)]

            if angle2 < angle1:
                angle2 += 2 * math.pi

            mid_angle = (angle1 + angle2) / 2

            # Draw boundary line
            start = ctx.camera.world_to_screen((cx, cy))
            end_x = cx + min(self.width, self.height) * 0.5 * math.cos(mid_angle)
            end_y = cy + min(self.width, self.height) * 0.5 * math.sin(mid_angle)
            end = ctx.camera.world_to_screen((end_x, end_y))

            pygame.draw.line(ctx.screen, (255, 0, 255), start, end, 2)

    def _draw_array(self, ctx: RenderContext, arr: np.ndarray) -> None:
        """Debug helper to visualize a 2D array as grayscale."""
        ts = self.ts
        mx = np.nanmax(arr)
        mn = np.nanmin(arr)
        if mx == mn:
            normalized = np.zeros_like(arr)
        else:
            normalized = (arr - mn) / (mx - mn)

        for tile, val in np.ndenumerate(normalized):
            if np.isnan(val):
                draw_tile(ctx, tile, ts, (200, 0, 0))
            else:
                gray = int(255 * val)
                draw_tile(ctx, tile, ts, (gray, gray, gray))

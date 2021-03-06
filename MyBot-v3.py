import hlt
import logging
from collections import OrderedDict

game = hlt.Game("RonaldinhoBot-V1")

logging.info("Starting RonaldinhoBot")

while True:
    game_map = game.update_map()
    command_queue = []
    team_ships = game_map.get_me().all_ships()

    for ship in team_ships:
        shipid = ship.id
        # if the ship is docked, skip the ship
        if ship.docking_status != ship.DockingStatus.UNDOCKED:
            continue

        # built in function from hlt
        entities_by_distance = game_map.nearby_entities_by_distance(ship)
        # we want ordered
        entities_by_distance = OrderedDict(sorted(entities_by_distance.items(), key=lambda t: t[0]))

        # we're iteratinf thro the entities_by_distance and if the
        # entity is a planet and isn't owned, we add it to our list
        # in order of the distance
        closest_empty_planets = [entities_by_distance[distance][0] for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and not entities_by_distance[distance][0].is_owned()]
        closest_enemy_ships = [entities_by_distance[distance][0] for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Ship) and entities_by_distance[distance][0] not in team_ships]

        if len(closest_empty_planets) > 0:
            target_planet = closest_empty_planets[0]
            if ship.can_dock(target_planet):
                command_queue.append(ship.dock(target_planet))
            else:
                navigate_command = ship.navigate(ship.closest_point_to(target_planet), game_map, speed = int(hlt.constants.MAX_SPEED), ignore_ships=False)
                if navigate_command:
                    command_queue.append(navigate_command)
        elif len(closest_enemy_ships) > 0:
            target_ship = closest_enemy_ships[0]
            navigate_command = ship.navigate(ship.closest_point_to(target_ship), game_map, speed = int(hlt.constants.MAX_SPEED), ignore_ships=False)
            if navigate_command:
                command_queue.append(navigate_command)

    game.send_command_queue(command_queue)
    # turn end

# game end
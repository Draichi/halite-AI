## we gonna basically run this games locally a bunch of times, 
# save whoever the winner was than train the model with this 
# wnners data

import hlt
import logging
## we want a dict that for sure retains our order
from collections import OrderedDict
import numpy as np
import random
import os

VERSION = 2

# how many features per entite that we want to track
HM_ENT_FEATURES = 5
PCT_CHANGE_CHANCE = 30
# 20 ~ 50
DESIRED_SHIP_COUNT = 20

game = hlt.Game("Ronaldinho{}".format(VERSION))
logging.info("RonaldinhoBot-{} Start".format(VERSION))

ship_plans = {}

# if the file exist we gonna delete it.
# every game we gonna save all of this players moves
# 2 scripts, p1 and p2. wich player wins, we gonna take
# this winning moves and use them to train
if os.path.exists("c{}_input.vec".format(VERSION)):
    os.remove("c{}_input.vec".format(VERSION))
if os.path.exists("c{}_out.vec".format(VERSION)):
    os.remove("c{}_out.vec".format(VERSION))

# this just finds the dictionary 
# key by their dictonarys values
def key_by_value(dictionary, value):
    for k,v in dictionary.items():
        if v[0] == value:
            return k
    # sometimes we not gonna have any value
    # so we put a -99 in their place,
    # otherwise we return just the key 
    # (the distance in most cases) 
    return -99

# we're asking to have 5 features but many times
# we wont own 5 planets, so it will pad it out
# the missing values
def fix_data(data):
    new_list = []
    last_known_idx = 0
    for i in range(HM_ENT_FEATURES):
        try:
            if i < len(data):
                last_known_idx=i
            new_list.append(data[last_known_idx])
        except:
            new_list.append(0)
    return new_list

while True:
    game_map = game.update_map()
    command_queue = []
    # team x enemys
    team_ships = game_map.get_me().all_ships()
    all_ships = game_map._all_ships()
    enemy_ships = [ship for ship in game_map._all_ships() if ship not in team_ships]
    # count
    my_ship_count = len(team_ships)
    enemy_ship_count = len(enemy_ships)
    all_ship_count = len(all_ships)
    # id 
    my_id = game_map.get_me().id
    # planet sizes
    empty_planet_sizes = {}
    our_planet_sizes = {}
    enemy_planet_sizes = {}
    # iterate trou all planets and save 
    # as empty/our/enemy
    for p in game_map.all_planets():
        radius = p.radius
        if not p.is_owned():
            empty_planet_sizes[radius] = p
        elif p.owner.id == my_id:
            our_planet_sizes[radius] = p
        elif p.owner.id != my_id:
            enemy_planet_sizes[radius] = p
    # how many planets each player have
    hm_our_planets = len(our_planet_sizes)
    hm_empty_planets = len(empty_planet_sizes)
    hm_enemy_planets = len(enemy_planet_sizes)
    # sorted give us small to big, [::-1] give big to small 
    empty_planet_keys = sorted([k for k in empty_planet_sizes])[::-1]
    our_planet_keys = sorted([k for k in our_planet_sizes])[::-1]
    enemy_planet_keys = sorted([k for k in enemy_planet_sizes])[::-1]
    # iterate tro ships
    for ship in game_map.get_me().all_ships():
        try:
            if ship.docking_status != ship.DockingStatus.UNDOCKED:
                # skip this ship
                continue
            # decide if gonna change
            shipid = ship.id
            change = False
            if random.randint(1, 100) <= PCT_CHANGE_CHANCE:
                change = True
            # we grab entities by distance them sort by the key
            # closest to furtest 
            entities_by_distance = game_map.nearby_entities_by_distance(ship)
            entities_by_distance = OrderedDict(sorted(entities_by_distance.items(), key=lambda t: t[0]))
            # entities_by_distance[distance][0] is the actual value of this entitie
            closest_empty_planets = [entities_by_distance[distance][0] for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and not entities_by_distance[distance][0].is_owned()]
            closest_empty_planet_distances = [distance for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and not entities_by_distance[distance][0].is_owned()]
            # same for my planets
            closest_my_planets = [entities_by_distance[distance][0] for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and entities_by_distance[distance][0].is_owned() and (entities_by_distance[distance][0].owner.id == game_map.get_me().id)]
            closest_my_planets_distances = [distance for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and entities_by_distance[distance][0].is_owned() and (entities_by_distance[distance][0].owner.id == game_map.get_me().id)]
            # same for enemy planets
            closest_enemy_planets = [entities_by_distance[distance][0] for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and entities_by_distance[distance][0] not in closest_my_planets and entities_by_distance[distance][0] not in closest_empty_planets]
            closest_enemy_planets_distances = [distance for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and entities_by_distance[distance][0] not in closest_my_planets and entities_by_distance[distance][0] not in closest_empty_planets]
            # largest distances
            largest_empty_planet_distances = []
            largest_our_planet_distances = []
            largest_enemy_planet_distances = []
            # at the start of the game we dont have any planet,
            # so we populate this with -99
            for i in range(HM_ENT_FEATURES):
                # empty
                try: largest_empty_planet_distances.append(key_by_value(entities_by_distance, empty_planet_sizes[empty_planet_keys[i]]))
                except:largest_empty_planet_distances.append(-99)
                # our
                try: largest_our_planet_distances.append(key_by_value(entities_by_distance, our_planet_sizes[our_planet_keys[i]]))
                except:largest_our_planet_distances.append(-99)
                # enemy
                try: largest_enemy_planet_distances.append(key_by_value(entities_by_distance, enemy_planet_sizes[enemy_planet_keys[i]]))
                except:largest_enemy_planet_distances.append(-99)
            # fix data will make shure that we have as 
            # many values as we're expecting
            entity_lists = [fix_data(closest_empty_planet_distances),
                            fix_data(closest_my_planets_distances),
                            fix_data(closest_enemy_planets_distances),
                            fix_data(closest_team_ships_distances),
                            fix_data(closest_enemy_ships_distances),
                            fix_data(empty_planet_keys),
                            fix_data(our_planet_keys),
                            fix_data(enemy_planet_keys),
                            fix_data(largest_empty_planet_distances),
                            fix_data(largest_our_planet_distances),
                            fix_data(largest_enemy_planet_distances)]
            input_vector = []
            # we gonna iterate tro each values
            # of that list of list, up to how many
            # features we're requaring, and append to input_vector
            # so, input value will have:
            # the 5 closest empty planet distances, 
            # the 5 closest my planets, 
            # the 5 closest enemy planets,
            # and so on...
            for i in entity_lists:
                for ii in i[:HM_ENT_FEATURES]:
                    input_vector.append(ii)
            # we also want few more stats:
            input_vector += [my_ship_count,
                             enemy_ship_count,
                             hm_our_planets,
                             hm_empty_planets,
                             hm_enemy_planets]
            # we already calculate all of our features, 
            # now we wanna calculate the output vector
            if my_ship_count > DESIRED_SHIP_COUNT:
                # if our ship count is greater than our desired,
                # we send them to dead (attack)
                output_vector = 3*[0]
                # ATTACK: [1,0,0]
                output_vector[0] = 1
                ship_plans[ship.id] = output_vector
            # if change is true and this ship doen't have a "plan"
            # we gonna pick a new "plan"
            elif change or ship.id not in ship_plans:
                output_vector = 3*[0]
                # random option between this 3 options will be a 1
                output_vector[random.randint(0,2)] = 1
                ship_plans[ship.id] = output_vector
            # if all elses fails
            else:
                # continue to execute existing plan
                output_vector = ship_plans[ship.id]
            # if the max values is the zeroth element (attack), we attack
            try:
                # ATTACK ENEMY SHIP #
                if np.argmax(output_vector) == 0:
                    #type: 0
                    #Find closest enemy ship, and attack!
                    if not isinstance(closest_enemy_ships[0], int):
                        navigate_command = ship.navigate(
                                    ship.closest_point_to(closest_enemy_ships[0]),
                                    game_map,
                                    speed=int(hlt.constants.MAX_SPEED),
                                    ignore_ships=False)
                        if navigate_command:
                            command_queue.append(navigate_command)
                # MINE ONE OF OUR PLANETS
                elif np.argmax(output_vector) == 1:
                    # type: 1
                    # Mine closest already-owned planet
                    if not isinstance(closest_my_planets[0], int):
                        target =  closest_my_planets[0]
                        # if the number of docked ships is less than the docking 
                        # opots, than we can dock
                        if len(target._docked_ship_ids) < target.num_docking_spots:
                            if ship.can_dock(target):
                                command_queue.append(ship.dock(target))
                            else:
                                navigate_command = ship.navigate(
                                            ship.closest_point_to(target),
                                            game_map,
                                            speed=int(hlt.constants.MAX_SPEED),
                                            ignore_ships=False)
                                if navigate_command:
                                    command_queue.append(navigate_command)
                        else:
                            #attack!
                            if not isinstance(closest_enemy_ships[0], int):
                                navigate_command = ship.navigate(
                                            ship.closest_point_to(closest_enemy_ships[0]),
                                            game_map,
                                            speed=int(hlt.constants.MAX_SPEED),
                                            ignore_ships=False)
                                if navigate_command:
                                    command_queue.append(navigate_command)
                    # if we dont have a closest own planet
                    # than we fallback and to the closest empty planet
                    elif not isinstance(closest_empty_planets[0], int):
                        target =  closest_empty_planets[0]
                        if ship.can_dock(target):
                            command_queue.append(ship.dock(target))
                        else:
                            navigate_command = ship.navigate(
                                        ship.closest_point_to(target),
                                        game_map,
                                        speed=int(hlt.constants.MAX_SPEED),
                                        ignore_ships=False)
                            if navigate_command:
                                command_queue.append(navigate_command)
                    #attack!
                    elif not isinstance(closest_enemy_ships[0], int):
                        navigate_command = ship.navigate(
                                    ship.closest_point_to(closest_enemy_ships[0]),
                                    game_map,
                                    speed=int(hlt.constants.MAX_SPEED),
                                    ignore_ships=False)
                        if navigate_command:
                            command_queue.append(navigate_command)
                # FIND AND MINE AN EMPTY PLANET #
                elif np.argmax(output_vector) == 2:
                    # type: 2
                    # Mine an empty planet. 
                    if not isinstance(closest_empty_planets[0], int):
                        target =  closest_empty_planets[0]
                        if ship.can_dock(target):
                            command_queue.append(ship.dock(target))
                        else:
                            navigate_command = ship.navigate(
                                        ship.closest_point_to(target),
                                        game_map,
                                        speed=int(hlt.constants.MAX_SPEED),
                                        ignore_ships=False)
                            if navigate_command:
                                command_queue.append(navigate_command)
                    else:
                        #attack!
                        if not isinstance(closest_enemy_ships[0], int):
                            navigate_command = ship.navigate(
                                        ship.closest_point_to(closest_enemy_ships[0]),
                                        game_map,
                                        speed=int(hlt.constants.MAX_SPEED),
                                        ignore_ships=False)
                            if navigate_command:
                                command_queue.append(navigate_command)
            except Exception as e:
                logging.info(str(e))
            with open("c{}_input.vec".format(VERSION),"a") as f:
                f.write(str( [round(item,3) for item in input_vector] ))
                f.write('\n')
            with open("c{}_out.vec".format(VERSION),"a") as f:
                f.write(str(output_vector))
                f.write('\n')
        except Exception as e:
            logging.info(str(e))
    game.send_command_queue(command_queue)
    # TURN END
# GAME END
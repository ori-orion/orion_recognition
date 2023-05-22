"""
Author: Matthew Munks
Owner: Matthew Munks

Copied from the orion_semantic_map package. This will interface directly with the
pymongo database, hopefully allowing for a slightly streamlined pipeline.
"""

import pymongo
import pymongo.cursor
import pymongo.collection
import datetime
import rospy
from typing import Tuple, List;
import math;

import std_srvs.srv;

# import utils
# from utils import UID_ENTRY, SESSION_ID
SESSION_ID = "SESSION_NUM";
UID_ENTRY = "UID";
GLOBAL_TIME_STAMP_ENTRY = "global_timestamp";

PYMONGO_ID_SPECIFIER = "_id";

DEBUG = True;
DEBUG_LONG = False;

# The root for all things som related.
SERVICE_ROOT = "som/";
DELETE_ALL_SERVICE_NAME = SERVICE_ROOT + "delete_databases";

class MemoryManager:
    """
    The manager for the connecting to the pymongo database.
    """
    def __init__(self, root="localhost", port=62345, connect_to_current_latest=False):
        """
        Most of these are self explanatory - the pymongo database hosts itself in localhost on 
        a given port.

        connect_to_current_latest - Let's say there are multiple memory systems (or instances of
            this class) running on the robot at a given time. If this flag is True, then it will
            latch onto the previous session (hopefully the one started in the other rosnode) rather 
            than creating a new one. That does however imply a start up order for the rosnodes.
        """
        self.client = pymongo.MongoClient(root, port);
        # self.clear_db();
        self.database = self.client.database_test;

        self.collections:dict = {};

        #region Setting up the session stuff.
        session_log = self.database.session_log_coll;
        if session_log.estimated_document_count() > 0:
            #https://stackoverflow.com/questions/32076382/mongodb-how-to-get-max-value-from-collections
            # => .find().sort(...).limit(...) is actually quite efficient
            previous_session_cursor:pymongo.cursor.Cursor = session_log.find().sort(SESSION_ID, -1).limit(1);
            previous_session:list = list(previous_session_cursor);
            if connect_to_current_latest == True:
                rospy.wait_for_service(DELETE_ALL_SERVICE_NAME);
                rospy.sleep(0.1);   # Make sure the service is set up. The race condition should be solved by wait_for_service, but this is just for robustness.
                self.current_session_id = previous_session[0][SESSION_ID];
            else:
                self.current_session_id = previous_session[0][SESSION_ID] + 1;
        else:
            self.current_session_id = 1;
        print("Session:", self.current_session_id);

        if self.current_session_id == 1 or connect_to_current_latest==False:
            session_log.insert_one({
                SESSION_ID: self.current_session_id,
                GLOBAL_TIME_STAMP_ENTRY: datetime.datetime.now()
            });
        #endregion

        # We only want to set up the services if we are on the root directory. 
        if connect_to_current_latest == False:
            self.setup_services();

    def addCollection(self, collection_name:str) -> pymongo.collection.Collection:
        """
        Adds a collection to the database.
        If the collection already exists, this acts as a get funciton.
        """
        self.collections[collection_name] = self.database[collection_name];
        return self.collections[collection_name];

    def clear_db(self, database_name='database_test'):
        """
        Gets rid of the entire database.
        """
        self.client.drop_database(database_name);

    def clear_database_ROS_server(self, srv_input:std_srvs.srv.EmptyRequest):
        """
        ROS entrypoint for deleting the entire database.
        """
        self.clear_db();
        return std_srvs.srv.EmptyResponse();

    def setup_services(self):
        """
        Function to setup all the services.
        """
        rospy.Service(DELETE_ALL_SERVICE_NAME, std_srvs.srv.Empty, self.clear_database_ROS_server);


class PerceptionInterface:
    """
    This is the interface with perception into the memory system.
    """
    def __init__(self, memory_manager:MemoryManager):
        self.memory_manager:MemoryManager = memory_manager;

        # The name of the som collection for objects is "objects"
        self.object_collection = self.memory_manager.addCollection("objects");

        # The name of the som collection for humans is "humans"
        self.human_collection = self.memory_manager.addCollection("humans");

        # This might be too close, but it should be ok.
        # NOTE: ROSParam?
        self.consistent_obj_distance_threshold = 0.02;
        pass;
    
    def getTimeDict(self) -> dict:
        time_of_creation = rospy.Time.now();
        time_of_creation_dict = {
            "secs" : time_of_creation.secs,
            "nsecs" : time_of_creation.nsecs }
        return time_of_creation_dict;

    def queryForObj_FindMatch(self, obj_class:str, observation_batch:int, position:Tuple[float]):
        """
        If we have a re-observation of an object, we want to have a quick check to
        see if we've seen this object before.

        Assuming there is only one detection per object, we will also want to ensure that we're not adding an object
        in the same batch for the sake of consistency.    
        """
        # We are looking for items of
        #   the same session id,
        #   the same class and, 
        #   observations in the past, rather than current ones. 
        prev_detections:List[dict] = list( self.object_collection.find({
            "HEADER"                    : { "session_num" : self.memory_manager.current_session_id },
            "class_"                    : obj_class,
            'last_observation_batch'    : { "$lt" : observation_batch }
        }));

        updating = None;
        smallest_distance = self.consistent_obj_distance_threshold;

        for detection in prev_detections:
            pos_dict:dict = detection["obj_position"]["position"];
            resp_pos_delta:Tuple[float] = (pos_dict['x']-position[0], pos_dict['y']-position[1], pos_dict['z']-position[2]);
            distance = 0;
            for pos_delta in resp_pos_delta:
                distance += pos_delta**2;
            distance = math.sqrt(distance);
            if distance < smallest_distance:
                updating = detection[PYMONGO_ID_SPECIFIER];
                smallest_distance = distance;
        
        if updating != None:
            # We need to update!
            pass;
        else:
            # Add a new item.
            pass;

        pass;

    def createNewObject(
            self, 
            obj_class:str, 
            observation_batch:int, 
            category:str,
            position:Tuple[float]) -> str:
        """
        Creates a new instance of an object.

        returns the uid of the object in the database.
        """



        time_of_creation_dict = self.getTimeDict();
        adding_dict = {
            "HEADER" : {
                "SESSION_NUM" : self.memory_manager.current_session_id
            },
            "class_" : obj_class,
            # "colour" : ...,
            "last_observation_batch" : observation_batch,
            "num_observations" : 1,
            "category" : category,
            "obj_position" : {
                "position" : {
                    "x" : position[0], "y" : position[1], "z" : position[2]
                },
                "orientation" : {
                    "x" : 0, "y" : 0, "z" : 0, "w" : 1
                }
            },
            # "pickupable" : ...,
            "picked_up" : False,
            "first_observed_at" : time_of_creation_dict,
            "last_observed_at" : time_of_creation_dict,
            # "size" : ...,
        }

        result = self.object_collection.insert_one(adding_dict);
        result_id:pymongo.collection.ObjectId = result.inserted_id;
        return str(result_id);
    
    def updateObject(
            self,
            uid_updating:str,
            current_batch_num:int,
            position:Tuple[float]) -> None:
        """
        Updates a given object (based on a uid reference)
        This is what should be done through object tracking.

        Assumptions:
            Object class stays the same, as does the object category.
        """

        time_of_observation_dict = self.getTimeDict();
        update_dict = {
            "$inc" : {"num_observations" : 1},
            "$set" : {
                "last_observed_at" : time_of_observation_dict,
                "last_observation_batch" : current_batch_num,
                "obj_position" : {
                    "position" : {
                        "x" : position[0], "y" : position[1], "z" : position[2]
                    },
                    "orientation" : {
                        "x" : 0, "y" : 0, "z" : 0, "w" : 1
                    }
                }
            }
        }

        self.object_collection.find_one_and_update(
            filter={"_id":pymongo.collection.ObjectId(uid_updating)},
            update=update_dict);

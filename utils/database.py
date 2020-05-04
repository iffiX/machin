import logging
import pymongo as mon
import pymongo.errors as mon_err

from urllib.parse import quote_plus
from typing import List, Mapping, Union, Any

from utils.logging import default_logger


class Database:
    """
    Brief;
        Base class for MongoDB collection manipulators.
    Args:
        host:
        port:
        username:
        password:
        database:
        max_timeout:
        loggers:
    """
    def __init__(self,
                 host: str,
                 port: int,
                 username: str,
                 password: str,
                 database: str,
                 max_timeout: int,
                 loggers: Union[str, list]) -> None:
        uri = "mongodb://%s:%s@%s:%d/%s" % (quote_plus(username), quote_plus(password), host, port, database)
        self.client = mon.MongoClient(uri, serverSelectionTimeoutMS=max_timeout)
        self.loggers = []
        if isinstance(loggers, str):
            self.loggers.append(logging.getLogger(loggers))
        elif isinstance(loggers, list):
            for l in loggers:
                self.loggers.append(logging.getLogger(l))
        try:
            self.client.server_info()
        except mon_err.ServerSelectionTimeoutError as _:
            default_logger.error("An error occurred while connecting to mongoDB")

    def log(self, msg: str, level=logging.INFO):
        """
        :param msg:
        :param level:
        :return:
        """
        for l in self.loggers:
            l.log(level, msg)


class ArtistDatabase(Database):
    def __init__(self,
                 host: str,
                 port: int,
                 username: str,
                 password: str,
                 database: str = "e621",
                 collection: str = "artist",
                 max_timeout=1000,
                 loggers: Union[str, list] = "default_logger",
                 **kwargs) -> None:
        # TODO: add error check for getting database and collection
        Database.__init__(self, host, port, username, password, database, max_timeout, loggers)
        self.database = self.client.get_database(name=database)
        self.collection = self.database.get_collection(name=collection)
        self.config = kwargs

    def insert(self,
               name: str,
               art_id: int,
               alias: List[str],
               url: List[str],
               active: bool) -> Any:
        doc = {
            "name": name,
            "art_id": art_id,
            "alias": alias,
            "url": url,
            "active": active
        }
        try:
            doc_id = self.collection.replace_one({"art_id": art_id}, doc, upsert=True)
        except mon_err.OperationFailure as e:
            self.log("MongoDB Err: %d, details: %s" % (e.code, e.details), logging.ERROR)
            return None
        return doc_id

    def get(self, name: str=None, id: int=None) -> List[Mapping]:
        result = []
        try:
            if name is not None:
                q_result = self.collection.find_many({"name": name})
                for artist in q_result:
                    result.append(artist)
            elif id is not None:
                q_result = self.collection.find_many({"name": name})
                result.append(q_result)
        except mon_err.OperationFailure as e:
            self.log("MongoDB Err: %d, details: %s" % (e.code, e.details), logging.ERROR)
            return [{}]
        return result


class FileDatabase(Database):
    def __init__(self,
                 host: str,
                 port: int,
                 username: str,
                 password: str,
                 database: str = "e621",
                 collection: str = "images",
                 max_timeout=1000,
                 loggers: Union[str, list] = "default_logger",
                 **kwargs) -> None:
        # TODO: add error check for getting database and collection
        Database.__init__(self, host, port, username, password, database, max_timeout, loggers)
        self.database = self.client.get_database(name=database)
        self.collection = self.database.get_collection(name=collection)
        self.config = kwargs

    def insert(self,
               id: int,
               url: str,
               file_name: str,
               tags: List,
               statistics: Mapping):
        doc = {
            "id": id,
            "url": url,
            "filename": file_name,
            "tags": tags,
            "statistics": statistics
        }
        try:
            doc_id = self.collection.replace_one({"filename": file_name}, doc, upsert=True)
        except mon_err.OperationFailure as e:
            self.log("MongoDB Err: %d, details: %s" % (e.code, e.details), logging.ERROR)
            return None
        return doc_id
from email.mime import base
from secrets import token_bytes
import pyodbc
import config_public as config
import logging
from API_Azure import AzureOpenAIPlatformService
from config_public import (
	ZILLIZ_URI, ZILLIZ_API_KEY, ZILLIZ_DB_NAME,
	PappsDB_UserID, PappsDB_PW, PappsDB_Server
)
from API_Zilliz import ZillizVectorStorageService
from vector_sync.data_models import Dynamics_ProjectBriefs, Dynamics_Experience
from azure.identity import ClientSecretCredential
from decimal import Decimal
import datetime
import os
import struct
import time

class DynamicsSync:
	# --- Config for scalable multi-query processing ---
	SQL_QUERIES_CONFIG = [
		{
			"name": "ProjectBriefs",
			"sql": "SELECT * FROM [dbo].[ProjectBriefsFiltered]",
			"dataclass": Dynamics_ProjectBriefs,
			"vector_db": "ProjectBriefs",
			"vector_field": "Brief",
			"field_map": [
				"Brief", "ProjectNumber", "BriefName", "Created", "LongName", "ShortName",
				"HeaderOrgRegion", "HeaderLocation", "HeaderDept", "Client", "ProjectManager",
				"ProjectURL", "BriefURL"
			]
		},
		{
			"name": "Experiences",
			"sql": "SELECT * FROM [ai].[ExperienceFiltered]",
			"dataclass": Dynamics_Experience,
			"vector_db": "Experiences",
			"vector_field": "EmpDescription",
			"field_map": [
				"ProjectNumber", "LongName", "ShortName", "Location", "Department", "Region",
				"ProjectURL", "EmpName", "EmpEmail", "WorkPhone", "EmpNumber", "Role",
				"Hours", "StartDate", "EndDate", "EmpDescription", "ExperienceURL"
			]
		},
	]

	def __init__(self):
		self.openai = AzureOpenAIPlatformService()
		self._zilliz = ZillizVectorStorageService(ZILLIZ_URI, ZILLIZ_API_KEY, ZILLIZ_DB_NAME)
		self.PappsDB_Server = PappsDB_Server
		self.PappsDB_UserID = PappsDB_UserID
		self.PappsDB_PW = PappsDB_PW

	def process_all_queries(self):
		try:
			for query_cfg in self.SQL_QUERIES_CONFIG:
				try:
					records = self.fetch_records_for_config(query_cfg)
					logging.info(f"Fetched {len(records)} records for {query_cfg['name']}")
				except Exception as e:
					logging.error(f"Failed to fetch records for {query_cfg['name']}: {e}")
					raise
				# Delete all vectors in the respective database/collection before upserting new ones
				try:
					self._zilliz.zilliz_delete_all_vectors(query_cfg["vector_db"])
					logging.info(f"Deleted all vectors in {query_cfg['vector_db']} collection before upsert.")
				except Exception as e:
					logging.error(f"Failed to delete vectors in {query_cfg['vector_db']}: {e}")
					raise

				# --- Bulk Embedding and Batch Upsert ---
				EMBEDDING_BATCH_SIZE = 15
				ZILLIZ_BATCH_SIZE = 30
				vector_field = query_cfg["vector_field"]
				total_records = len(records)
				rows: list = []
				for i in range(0, total_records, EMBEDDING_BATCH_SIZE):
					batch_records = records[i:i+EMBEDDING_BATCH_SIZE]
					texts_to_embed = [getattr(rec, vector_field, "") for rec in batch_records]
					try:
						vectors = self.openai.generate_embeddings_bulk(texts_to_embed)
						time.sleep(0.1)
					except Exception as e:
						logging.error(f"Bulk embedding failed for {query_cfg['name']} batch {i}-{i+EMBEDDING_BATCH_SIZE}: {e}")
						raise
					for rec, vector in zip(batch_records, vectors):
						base = {field: sanitize(getattr(rec, field, None)) for field in query_cfg["field_map"]}
						base["vector"] = [float(x) for x in vector]
						rows.append(base)
						# Batch upsert to Zilliz
						if len(rows) >= ZILLIZ_BATCH_SIZE:
							try:
								self._zilliz.zilliz_insert_chunk_rows(query_cfg["vector_db"], rows)
								logging.info(f"Inserted {len(rows)} rows into {query_cfg['vector_db']} collection (intermediate batch).")
								rows = []
							except Exception as e:
								logging.error(f"Failed to insert rows into {query_cfg['vector_db']}: {e}")
								raise
				# Insert any remaining rows
				if rows:
					try:
						self._zilliz.zilliz_insert_chunk_rows(query_cfg["vector_db"], rows)
						logging.info(f"Inserted {len(rows)} rows into {query_cfg['vector_db']} collection (final batch).")
					except Exception as e:
						logging.error(f"Failed to insert rows into {query_cfg['vector_db']}: {e}")
						raise
				logging.info(f"Upserted {total_records} records into {query_cfg['vector_db']} collection.")
		except Exception as e:
			logging.exception(f"process_all_queries failed: {e}")
			raise

	def fetch_records_for_config(self, query_cfg):
		try:
			# SQL Server authentication details
			sql_server = self.PappsDB_Server
			sql_database = "bgeazrpowerappsDB"
			driver = "ODBC Driver 18 for SQL Server"
			sql_username = self.PappsDB_UserID
			sql_password = self.PappsDB_PW

			# Build connection string (SQL Server Authentication)
			conn_str = (
				f"DRIVER={{{driver}}};"
				f"SERVER={sql_server};"
				f"DATABASE={sql_database};"
				f"UID={sql_username};"
				f"PWD={sql_password};"
				f"Encrypt=yes;"
				f"TrustServerCertificate=yes;"
			)

			records = []
			with pyodbc.connect(conn_str) as conn:
				cursor = conn.cursor()
				cursor.execute(query_cfg["sql"])
				for row in cursor.fetchall():
					kwargs = {field: getattr(row, field, None) for field in query_cfg["field_map"]}
					record = query_cfg["dataclass"](**kwargs)
					records.append(record)
			return records
		except Exception as e:
			logging.error(f"Error fetching records for config {query_cfg.get('name', 'unknown')}: {e}")
			raise


def sanitize(val):
    if val is None:
        return None
    if isinstance(val, (datetime.date, datetime.datetime)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, (bytes, bytearray)):
        return val.decode(errors="ignore")
    return val
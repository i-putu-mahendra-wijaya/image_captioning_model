from typing import List, Dict, Literal, Optional

from google.cloud import bigquery as bq
from google.cloud.bigquery.enums import SqlTypeNames
from google.cloud.bigquery.table import TimePartitioningType
from google.cloud.exceptions import NotFound
import pandas as pd
import datetime
from time import time
from dataclasses import dataclass
from pathlib import Path
from yaml import safe_load

from src.image_captioning_model.GCP.CredentialAccessor.CredentialAccessor import CredentialAccessor

@dataclass
class WriteSpecification:

    """
    Specification for writing data into a target BigQuery dataset.

    This dataclass defines configuration parameters for executing a write
    operation to BigQuery. It is typically used to specify project settings,
    target dataset, and write behavior when loading data.

    Parameters
    ----------
    project_name : str
        A human-readable name of the project associated with the write operation.
        This value may be used for logging, monitoring, or audit purposes.

    project_id : str
        The Google Cloud project ID where the target BigQuery dataset resides.
        This is required to ensure the write operation is executed in
        the correct GCP environment.

    location : str
        The Google Cloud region where the BigQuery dataset resides.
        This is required to ensure the write operation is executed in
        the correct BigQuery location.

    target_dataset : str
        The BigQuery dataset name where the data should be written.
        The dataset must already exist in the specified project.

    write_disposition : {"WRITE_TRUNCATE", "WRITE_APPEND", "WRITE_EMPTY"}
        Defines the write behavior when data already exists in the destination.

        - ``"WRITE_TRUNCATE"``: Overwrites the entire destination table.
        - ``"WRITE_APPEND"``: Appends data to the destination table.
        - ``"WRITE_EMPTY"``: Fails if the destination table is not empty.

    Raises
    ------
    ValueError
        If an unsupported ``write_disposition`` value is provided.

    Examples
    --------
    >>> write_spec = WriteSpecification(
    ...     project_name="Production Odoo ETL",
    ...     project_id="disco-song-343611",
    ...     location="asia-southeast2",
    ...     target_dataset="odoo_v2_raw",
    ...     write_disposition="WRITE_APPEND",
    ... )
    """

    project_name: str
    project_id: str
    location: str
    target_dataset: str
    write_disposition: Literal["WRITE_TRUNCATE", "WRITE_APPEND", "WRITE_EMPTY"]

    def __post_init__(
            self
    ) -> None:
        allowed_write_dispositions = ["WRITE_TRUNCATE", "WRITE_APPEND", "WRITE_EMPTY"]

        if self.write_disposition not in allowed_write_dispositions:
            raise ValueError(f"Unsupported write disposition: {self.write_disposition}. Allowed values: {allowed_write_dispositions}")


class YamlSpecParser:

    """
    Parser for YAML-based BigQuery write specifications.

    This class reads a YAML file containing one or more BigQuery write
    specifications and converts the first entry under the
    ``"write_specs"`` key into a :class:`WriteSpecification` object.

    The YAML file is expected to define a mapping with configuration
    parameters such as the allowed write dispositions and a list of
    write specifications. A typical structure is:

    .. code-block:: yaml

        allowed_write_dispositions:
          - WRITE_TRUNCATE
          - WRITE_APPEND
          - WRITE_EMPTY

        write_specs:
          - project_name: "agriaku-dwh-prod"
            project_id: "disco-song-343611"
            target_dataset: "odoo_v2_raw"
            write_disposition: "WRITE_APPEND"

    The parser extracts the first element of the ``write_specs`` list and
    instantiates a :class:`WriteSpecification` from it. Validation of the
    ``write_disposition`` value is delegated to
    :meth:`WriteSpecification.__post_init__`, which enforces that only
    supported dispositions are accepted (for example,
    ``"WRITE_TRUNCATE"``, ``"WRITE_APPEND"``, or ``"WRITE_EMPTY"``).

    Parameters
    ----------
    yaml_spec_path : Path
        Filesystem path pointing to the YAML specification file.

    Attributes
    ----------
    yaml_spec_path : Path
        Location of the YAML specification file.
    write_spec_dict : Dict
        Raw dictionary representing the selected write specification
        loaded from the YAML file.
    write_spec : WriteSpecification
        Parsed write specification object derived from ``write_spec_dict``.
        This object can be passed to BigQuery-related components to
        configure write operations.

    Raises
    ------
    KeyError
        If the ``"write_specs"`` key is missing or does not contain at
        least one specification.
    ValueError
        If the ``write_disposition`` value is unsupported and validation
        in :class:`WriteSpecification` fails.

    Examples
    --------
    >>> from pathlib import Path
    >>> parser = YamlParser(Path("./odoo_bigquery_write_specification.yaml"))
    >>> parser.write_spec.project_id
    'disco-song-343611'
    >>> parser.write_spec.write_disposition
    'WRITE_APPEND'
    """

    def __init__(
            self,
            yaml_spec_path: Path
    ) -> None:

        self.yaml_spec_path: Path = yaml_spec_path

        with open(self.yaml_spec_path, "r") as spec_handler:
            yaml_content_list_dict: List[Dict] = safe_load(spec_handler)
            self.write_spec_dict: Dict = yaml_content_list_dict.get("write_specs", [])[0]


        self.write_spec: WriteSpecification = WriteSpecification(
            **self.write_spec_dict
        )




class DaoBigQuery:

    """
    Data Access Object (DAO) for interacting with Google BigQuery.

    This class provides a high-level interface for common BigQuery operations,
    including creating a client, checking table existence, executing SQL queries,
    reading query results into pandas DataFrames, and loading data from
    DataFrames into BigQuery tables.

    The class relies on a :class:`CredentialAccessor` instance to obtain
    GCP service account credentials, the default project ID, and the target
    location for BigQuery operations.

    Parameters
    ----------
    credential_accessor : CredentialAccessor
        An object that provides access to the GCP service account credentials,
        project ID, and location used for BigQuery operations.

    Attributes
    ----------
    credential_accessor : CredentialAccessor
        Reference to the credential accessor provided during initialization.
    mygbq : google.cloud.bigquery.client.Client
        BigQuery client used to perform all interactions with BigQuery.

    Examples
    --------
    >>> dao = DaoBigQuery(credential_accessor)
    >>> df = dao.read_from_query("SELECT * FROM `my_dataset.my_table` LIMIT 10")
    >>> dao.write_to_bigquery(
    ...     data_df=df,
    ...     schema_name="my_dataset",
    ...     table_name="my_table_copy",
    ...     write_disposition="WRITE_TRUNCATE",
    ... )
    """

    def __init__(
            self,
            credential_accessor: CredentialAccessor
    ) -> None:

        """
        Initialize the BigQuery DAO and create a client.

        Parameters
        ----------
        credential_accessor : CredentialAccessor
            Object providing the GCP service account credentials, project ID,
            and location for BigQuery operations.

        Notes
        -----
        Upon initialization, this method immediately creates a BigQuery client
        via :meth:`create_client` and assigns it to :attr:`mygbq`.
        """

        self.credential_accessor: CredentialAccessor = credential_accessor
        self.mygbq: bq.Client = self.create_client()


    def create_client(
            self
    ) -> bq.Client:

        """
        Create a BigQuery client using the provided credentials and project ID.

        Returns
        -------
        google.cloud.bigquery.client.Client
            A configured BigQuery client instance.

        Notes
        -----
        The client is created using the service account credentials and project
        ID exposed by :attr:`credential_accessor`. This method is typically
        invoked internally during initialization.
        """

        return bq.Client(
            credentials=self.credential_accessor.gcp_sa_credentials,
            project=self.credential_accessor.project_id
        )

    def is_exists(
            self,
            schema_name: str,
            table_name: str
    ) -> bool:

        """
        Check whether a BigQuery table exists.

        Parameters
        ----------
        schema_name : str
            Name of the BigQuery dataset (schema) to check.
        table_name : str
            Name of the table within the given dataset.

        Returns
        -------
        bool
            ``True`` if the table exists, ``False`` otherwise.

        Notes
        -----
        This method attempts to fetch the table metadata. If BigQuery raises a
        :class:`google.cloud.exceptions.NotFound` error, the method returns
        ``False``; otherwise, it returns ``True``.
        """

        print(f"Checking if table `{schema_name}.{table_name}` exists...")

        try:
            rst = self.mygbq.get_table(f"{schema_name}.{table_name}")
            print(f"Table `{schema_name}.{table_name}` exists.")
            return True

        except NotFound:
            print(f"Table `{schema_name}.{table_name}` does not exist.")
            return False

    def execute_query(
            self,
            query: str
    ) -> bq.QueryJob:
        """
        Execute a SQL query on BigQuery.

        Parameters
        ----------
        query : str
            SQL query string to execute.

        Returns
        -------
        google.cloud.bigquery.job.QueryJob
            The BigQuery QueryJob object representing the executed query.

        Notes
        -----
        This method blocks until the query has finished running by calling
        :meth:`QueryJob.result`. If you need asynchronous behavior, you may
        modify this to return the job before waiting on the result.
        """

        print(f"Executing query: {query}")

        query_job: bq.QueryJob = self.mygbq.query(query)
        query_job.result()

        return query_job

    def read_from_query(
            self,
            query: str
    ) -> pd.DataFrame:

        """
        Execute a SQL query and return the result as a pandas DataFrame.

        Parameters
        ----------
        query : str
            SQL query string to execute.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the query results.

        Notes
        -----
        Internally, this method calls :meth:`execute_query` to run the query
        and then converts the resulting :class:`QueryJob` into a DataFrame
        using :meth:`QueryJob.to_dataframe`.
        """

        query_job: bq.QueryJob = self.execute_query(query)

        result_df: pd.DataFrame = query_job.to_dataframe()

        return result_df

    def write_to_bigquery(
            self,
            data_df: pd.DataFrame,
            schema_name: str,
            table_name: str,
            write_disposition: Literal["WRITE_TRUNCATE", "WRITE_APPEND", "WRITE_EMPTY"] = "WRITE_TRUNCATE",
            time_partition_type: Optional[Literal["DAY", "HOUR", "MONTH", "YEAR"]] = None,
            time_partitioning_field: Optional[str] = None,
            clustering_fields: Optional[List[str]] = None,
            num_retries: int = 3
    ) -> None:

        """
        Write a pandas DataFrame to a BigQuery table.

        This method creates a BigQuery schema from the DataFrame dtypes,
        optionally configures time partitioning and clustering, and then uses
        :func:`load_table_from_dataframe` to load the data into the specified
        table.

        By default, the write operation uses ``"WRITE_TRUNCATE"`` as the
        write disposition and automatically adds a ``prc_dt`` column to record
        the load date.

        Parameters
        ----------
        data_df : pandas.DataFrame
            The DataFrame to write to BigQuery. A ``prc_dt`` column will be
            added to this DataFrame to store the processing date.
        schema_name : str
            Name of the target BigQuery dataset (schema).
        table_name : str
            Name of the target BigQuery table.
        write_disposition : {"WRITE_TRUNCATE", "WRITE_APPEND", "WRITE_EMPTY"}, optional
            BigQuery write disposition for the load job.

            - ``"WRITE_TRUNCATE"`` (default): Overwrite the destination table.
            - ``"WRITE_APPEND"``: Append to the destination table.
            - ``"WRITE_EMPTY"``: Fail if the destination table is not empty.

        time_partition_type : {"DAY", "HOUR", "MONTH", "YEAR"}, optional
            Type of time partitioning to apply to the table. If provided,
            :paramref:`time_partitioning_field` must also be specified.
        time_partitioning_field : str, optional
            Name of the column to use as the time partitioning field. Required
            when :paramref:`time_partition_type` is not ``None``.
        clustering_fields : list of str, optional
            List of column names to use as clustering keys. If provided, the
            list must contain at most 4 fields, and all fields must exist in
            :paramref:`data_df`.
        num_retries : int, default 3
            Number of retries for the BigQuery load job in case of transient
            errors.

        Raises
        ------
        ValueError
            If more than 4 clustering fields are provided, if any clustering
            field is not present in the DataFrame columns, if an invalid
            time partition type is specified, or if a time partition type is
            provided without a corresponding partitioning field.

        Notes
        -----
        - A ``prc_dt`` column of type ``datetime64[ns]`` is added to the
          DataFrame before loading, representing the current date.
        - The BigQuery schema is inferred from the DataFrame dtypes using a
          predefined mapping to :class:`SqlTypeNames`.
        - The load job uses the project ID and location provided by
          :attr:`credential_accessor`.

        Examples
        --------
        >>> dao = DaoBigQuery(credential_accessor)
        >>> dao.write_to_bigquery(
        ...     data_df=df,
        ...     schema_name="odoo_v2_raw",
        ...     table_name="account_move_line",
        ...     write_disposition="WRITE_APPEND",
        ...     time_partition_type="DAY",
        ...     time_partitioning_field="prc_dt",
        ...     clustering_fields=["partner_id"],
        ... )
        """

        # record current time stamp
        current_time_stamp: str = datetime.datetime.now()

        data_df["prc_dt"] = current_time_stamp

        data_df = data_df.astype(
            {
                "prc_dt": "datetime64[ns]"
            }
        )

        column_names: List[str] = data_df.columns.tolist()

        if clustering_fields is not None:
            if len(clustering_fields) > 4:
                raise ValueError(f"{len(clustering_fields)} clustering fields specified, exceeding the limit of 4.")

            if all([each_clustering_field in data_df.columns for each_clustering_field in clustering_fields]) is False:
                raise ValueError(f"Clustering fields {clustering_fields} are not in the dataframe columns {data_df.columns}")


        valid_partition_type_: List[str] = ["DAY", "HOUR", "MONTH", "YEAR"]
        if time_partition_type is not None:
            if time_partition_type not in valid_partition_type_:
                raise ValueError(f"Invalid time partition type: {time_partition_type} | Valid types: {valid_partition_type_}")

            if time_partitioning_field is None:
                raise ValueError("time_partitioning_field must be specified when time_partition_type is specified")


        list_col_datatypes: List[str] = [str(each_emt) for each_emt in data_df.dtypes.tolist()]

        conversion_dict: Dict = {
            "object": SqlTypeNames.STRING,
            "geography": SqlTypeNames.GEOGRAPHY,
            "int64": SqlTypeNames.INT64,
            "float64": SqlTypeNames.FLOAT,
            "bool": SqlTypeNames.BOOL,
            "datetime64[ns]": SqlTypeNames.DATETIME,
            "timedelta[ns]": SqlTypeNames.INTERVAL,
            "category": SqlTypeNames.STRING
        }

        list_datatype_gbq: List[SqlTypeNames] = [
            conversion_dict[each_type] for each_type in list_col_datatypes
        ]

        bq_modes: List[str] = ["NULLABLE" for _ in range(len(column_names))]

        bq_partition_key: Optional[bq.TimePartitioning] = None

        if time_partition_type is not None:

            bq_time_partitioning_type_: TimePartitioningType = getattr(bq.TimePartitioningType, time_partition_type)

            bq_partition_key = bq.TimePartitioning(
                type_=bq_time_partitioning_type_,
                field=time_partitioning_field
            )

        tic: float = time()
        schema_: List[bq.SchemaField] = [
            bq.SchemaField(
                name=each_col,
                field_type=each_type,
                mode=each_mode
            )
            for each_col, each_type, each_mode in zip(column_names, list_datatype_gbq, bq_modes)
        ]

        load_job: bq.LoadJob = self.mygbq.load_table_from_dataframe(
            dataframe=data_df,
            destination=f"{schema_name}.{table_name}",
            job_config=bq.LoadJobConfig(
                schema=schema_,
                write_disposition=write_disposition,
                time_partitioning=bq_partition_key,
                clustering_fields=clustering_fields
            ),
            num_retries=num_retries,
            project=self.credential_accessor.project_id,
            location=self.credential_accessor.location
        )

        load_job.result() # wait for the load job to complete

        toc: float = time()

        print(f"[SUCCESS] Data was successfully inserted into `{schema_name}.{table_name}` in {toc - tic:.2f} seconds")

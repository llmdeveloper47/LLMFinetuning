import logging

def logger_object():

    logging.basicConfig(
        level=logging.DEBUG,  # Set the lowest level to capture all messages
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='experiment.log',  # Log output to a file named experiment.log
        filemode='w'  # Overwrite the log file each run
    )

    logger = logging.getLogger('GPTDatasetBuilding')
    # Console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set level for the console output
    console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    # File handler for logging
    file_handler = logging.FileHandler('experiment.log')
    file_handler.setLevel(logging.DEBUG)  # Log detailed information to file
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Add both handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
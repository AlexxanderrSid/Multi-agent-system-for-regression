import warnings
import logging

OUTER_LOOP_STEPS  = 3
INNER_LOOP_STEPS  = 5
MAX_DEBUG         = 3
ENSEMBLE_ROUNDS   = 5
RANDOM_STATE      = 42

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('MAS')

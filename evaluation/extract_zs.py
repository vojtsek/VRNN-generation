import sys
from .commons import TurnRecord

if __name__ == '__main__':
    records = []
    fn = sys.argv[1]
    role = sys.argv[2]
    TurnRecord.parse(fn, records, {}, role)
    for rec in records:
        print([r[1] for r in rec.prior_z_vector])

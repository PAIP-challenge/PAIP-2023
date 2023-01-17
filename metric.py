import pandas as pd
import pingouin as pg

gt_TC_fn = '/workspace/gt.csv'
pred_TC_fn = '/workspace/pred.csv'

pd_gt = pd.read_csv(gt_TC_fn)
pd_pred = pd.read_csv(pred_TC_fn)

if not (pd_pred['TC']<100).all() and (pd_pred['TC'>0]).all():
    print('All TC values should be computed between 0 and 100')
    exit()

pd_gt['tag'] = ['gt']*len(pd_gt)
pd_pred['tag'] = ['pred']*len(pd_pred)

pds = pd.concat((pd_gt, pd_pred))

# Intraclass Correlation
icc_table = pg.intraclass_corr(
                data=pds, targets='ID', raters='tag', ratings='TC'
            )

ICC_value = icc_table['ICC'][1] # ICC2

print(f'ICC_value: {ICC_value:04f}')

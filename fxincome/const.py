from enum import Enum, EnumMeta


class CustomEnumMeta(EnumMeta):
    def __new__(metacls, cls, bases, classdict):
        enum_class = super(CustomEnumMeta, metacls).__new__(metacls, cls, bases, classdict)
        enum_class._member_reverse_map = {v.value: v for v in enum_class.__members__.values()}
        return enum_class

    def __contains__(cls, member):
        if super(CustomEnumMeta, cls).__contains__(member):
            return True
        if isinstance(member, str):
            return member in cls._member_reverse_map
        return False

    def __getitem__(self, item):
        try:
            return super(CustomEnumMeta, self).__getitem__(item)
        except KeyError:
            return self._member_reverse_map[item]


class CustomEnum(str, Enum, metaclass=CustomEnumMeta):
    def __repr__(self):
        return "%s.%s" % (
            self.__class__.__name__, self._name_)


class COUPON_TYPE(CustomEnum):
    REGULAR = '附息'
    ZERO = '贴现'  # 贴现债券剩余期限不能超过1年
    DUE = '到期一次还本付息'  # TODO 现在算的到期还本付息债券不能超过1年


class ACCOUNT_TYPE(CustomEnum):
    OCI = 'OCI'
    TPL = 'TPL'
    AC = 'AC'


class CASHFLOW_TYPE(CustomEnum):
    Undelivered = 'Undelivered'
    Undelivered_Lastone = 'Undelivered_Lastone'
    History = 'History'
    All = 'All'


class CASHFLOW_VIEW_TYPE(CustomEnum):
    Raw = 'Raw'
    Agg = 'Agg'


class POSITION_GAIN_VIEW_TYPE(CustomEnum):
    Raw = 'Raw'
    Agg = 'Agg'


class DURARION_TYPE(CustomEnum):
    Macaulay = 'Macaulay'
    Modified = 'Modified'


class TBOND_PARAM:
    """
    Parameters for 16国债19(019547.SH)
    训练模型的参数。模型目标是预测16国债19收盘收益率的涨跌。
    参数主要是Features和Labels
    """

    ALL_FEATS = [
        # 原始features
        'date',
        'close',  # 收盘ytm
        'amount',  # 成交额，单位是元
        'ttm',  # Term to Maturity, 剩余期限，单位是年
        'fr007',  # 7天回购定盘利率
        't10y',  # 10年国债中债估值ytm
        'fr007_1y',  # 1年期 fr007 IRS CFETS收盘利率
        'fr007_5y',  # 5年期 fr007 IRS CFETS收盘利率
        'usdcny',  # 美元兑人民币汇率
        # 收盘ytm变种
        'pct_chg',
        'avg_chg_5',
        'avg_chg_20',
        'volaty',  # (ytm(low price)  - ytm(high price)) / ytm(close)
        # 流动性指标变种
        'fr007_chg_5',  # fr007 定盘利率 前5天均值变化率
        'fr007_1y_chg_5',  # 1年期 fr007 IRS CFETS收盘利率 前5天均值变化率
        # 收盘ytm与其他各种指标之间的差值
        'spread_t1y',
        'spread_t10y',
        'spread_fr007',
        'spread_fr007_1y',
        'spread_fr007_5y',
        'spread_usdcny',
        # 其他各种指标之间的差值
        'spread_fr007_5y_fr007_1y',  # 5年IRS - 1年IRS
        # 汇率变种
        'usdcny_chg_5',
        # label
        'future',
        'target'
    ]

    TRAIN_FEATS = [
        # 原始features
        'close',
        # 'amount',
        'ttm',
        # 'fr007',
        # 'fr007_5y',
        # 收盘ytm变种
        'pct_chg',
        # 'avg_chg_5',
        'avg_chg_20',
        # 'volaty',
        # 流动性指标变种
        'fr007_chg_5',
        # 收盘ytm与其他各种指标之间的差值
        # 'spread_t1y',
        'spread_t10y',
        # 'spread_fr007',
        # 'spread_fr007_5y',
        # 'spread_usdcny',
        # 其他各种指标之间的差值
        # 'spread_fr007_5y_fr007_1y',
        # 汇率变种
        'usdcny_chg_5'
    ]
    LABELS = ['target']


class MTM_PARAM:
    """
    Parameters for Mark to Market estimated value
    训练模型的参数。模型目标是预测中债估值收益率涨跌。
    参数主要是Features和Labels
    """

    ALL_FEATS = [
        # 原始features
        'date',
        'close',
        'fr007',
        'cdb10y',
        'fr0071y',
        'fr0075y',
        'usdcny',
        # 10年国债收益率变种
        'pct_chg',
        'avg_chg_5',
        'avg_chg_10',
        'avg_chg_20',
        'close_avg_5',
        # 流动性指标变种
        'fr007_chg_5',
        'fr0071y_chg_5',
        # 10年国债收益率与其他各种指标之间的差值
        'spread_t1y',
        'spread_cdb10y',
        'spread_fr007',
        'spread_fr0071y',
        'spread_fr0075y',
        'spread_usdcny',
        # 其他各种指标之间的差值
        'spread_fr0075y_fr0071y',
        # 汇率变种
        'usdcny_chg_5',
        # label
        'future',
        'target'
    ]

    TRAIN_FEATS = [
        # 原始features
        'close',
        'fr007',
        # 'cdb10y',
        # 'fr0071y',
        'fr0075y',
        # 'usdcny',
        # 10年国债收益率变种
        'pct_chg',
        'avg_chg_5',
        # 'avg_chg_10',
        'avg_chg_20',
        'close_avg_5',
        # 流动性指标变种
        'fr007_chg_5',
        # 'fr0071y_chg_5',
        # 10年国债收益率与其他各种指标之间的差值
        'spread_t1y',
        # 'spread_cdb10y',
        'spread_fr007',
        # 'spread_fr0071y',
        'spread_fr0075y',
        'spread_usdcny',
        # 其他各种指标之间的差值
        # 'spread_fr0075y_fr0071y',
        # 汇率变种
        'usdcny_chg_5',
    ]
    LABELS = ['target']


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
    REGULAR='附息'
    ZERO='贴现'  # 贴现债券剩余期限不能超过1年
    DUE='到期一次还本付息'  # TODO 现在算的到期还本付息债券不能超过1年

class ACCOUNT_TYPE(CustomEnum):
    OCI='OCI'
    TPL='TPL'
    AC='AC'
class CASHFLOW_TYPE(CustomEnum):
    Undelivered='Undelivered'
    Undelivered_Lastone='Undelivered_Lastone'
    History='History'
    All='All'
class CASHFLOW_VIEW_TYPE(CustomEnum):
    Raw='Raw'
    Agg='Agg'
class POSITION_GAIN_VIEW_TYPE(CustomEnum):
    Raw='Raw'
    Agg='Agg'
class DURARION_TYPE(CustomEnum):
    Macaulay='Macaulay'
    Modified='Modified'
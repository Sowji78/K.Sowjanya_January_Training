def add_order(order_id, orders=None):
    if orders is None:
        orders = []

    orders.append(order_id)
    return orders


# Sample calls
print(add_order(101))
print(add_order(102))
print(add_order(103))
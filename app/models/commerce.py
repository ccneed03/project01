"""
Demo commerce domain models.

These tables simulate the order / logistics / inventory / after-sales systems
for the controlled Agent demo. They are intentionally small and self-contained:
no external service integration and no multi-tenant columns.
"""
from datetime import datetime

from app.extensions import db


class Order(db.Model):
    """A customer order owned by a platform user."""

    __tablename__ = "orders"

    id = db.Column(db.Integer, primary_key=True)
    order_no = db.Column(db.String(64), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    status = db.Column(db.String(32), nullable=False, default="paid")  # paid, shipped, delivered, cancelled
    total_amount = db.Column(db.Float, nullable=False, default=0.0)

    recipient_name = db.Column(db.String(64))
    recipient_phone = db.Column(db.String(32))
    address = db.Column(db.String(512))

    paid_at = db.Column(db.DateTime)
    delivered_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    items = db.relationship(
        "OrderItem",
        back_populates="order",
        cascade="all, delete-orphan",
        order_by="OrderItem.id",
    )
    shipments = db.relationship(
        "Shipment",
        back_populates="order",
        cascade="all, delete-orphan",
        order_by="Shipment.id",
    )

    def __repr__(self):
        return f"<Order {self.order_no} user={self.user_id}>"


class OrderItem(db.Model):
    """A line item inside an order."""

    __tablename__ = "order_items"

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey("orders.id"), nullable=False, index=True)

    sku_id = db.Column(db.String(64), nullable=False, index=True)
    product_name = db.Column(db.String(256), nullable=False)
    category = db.Column(db.String(64), nullable=False, default="general")
    quantity = db.Column(db.Integer, nullable=False, default=1)
    unit_price = db.Column(db.Float, nullable=False, default=0.0)
    item_status = db.Column(db.String(32), nullable=False, default="normal")  # normal, returning, returned

    order = db.relationship("Order", back_populates="items")

    @property
    def subtotal(self) -> float:
        return round((self.unit_price or 0.0) * (self.quantity or 0), 2)

    def __repr__(self):
        return f"<OrderItem {self.sku_id} x{self.quantity}>"


class Shipment(db.Model):
    """A shipment associated with an order."""

    __tablename__ = "shipments"

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey("orders.id"), nullable=False, index=True)

    carrier = db.Column(db.String(64), nullable=False)
    tracking_no = db.Column(db.String(128), nullable=False)
    status = db.Column(db.String(32), nullable=False, default="created")  # created, in_transit, delivered
    last_event = db.Column(db.String(256))
    shipped_at = db.Column(db.DateTime)
    delivered_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)

    order = db.relationship("Order", back_populates="shipments")

    def __repr__(self):
        return f"<Shipment {self.tracking_no} {self.status}>"


class Inventory(db.Model):
    """Simple SKU stock table used by check_inventory."""

    __tablename__ = "inventory"

    id = db.Column(db.Integer, primary_key=True)
    sku_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    product_name = db.Column(db.String(256), nullable=False)
    stock = db.Column(db.Integer, nullable=False, default=0)
    reserved = db.Column(db.Integer, nullable=False, default=0)
    warehouse = db.Column(db.String(64), nullable=False, default="WH-1")
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f"<Inventory {self.sku_id} stock={self.stock}>"


class ReturnRequest(db.Model):
    """The result of a confirmed return action.

    ``idempotency_key`` is unique so a retried confirmation cannot create a
    second refund request.
    """

    __tablename__ = "return_requests"

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey("orders.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    item_ids_json = db.Column(db.JSON, nullable=False, default=list)
    reason = db.Column(db.String(512))
    refund_amount = db.Column(db.Float, nullable=False, default=0.0)
    status = db.Column(db.String(32), nullable=False, default="approved")  # approved, refunded, failed

    idempotency_key = db.Column(db.String(64), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    order = db.relationship("Order")

    def __repr__(self):
        return f"<ReturnRequest order={self.order_id} refund={self.refund_amount}>"


class Product(db.Model):
    """Authoritative product catalog.

    ``description`` / ``care_info`` / ``delivery_note`` / ``return_note`` are
    natural-language product texts that may be indexed into the vector store
    (same pipeline as the refund policy) so the agent can answer descriptive
    questions about materials, sizing, care and delivery.
    """

    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True)
    sku_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    product_name = db.Column(db.String(256), nullable=False)
    category = db.Column(db.String(64), nullable=False, default="general")
    brand = db.Column(db.String(64))

    # 产品说明（可入向量库的长文本）
    description = db.Column(db.Text, nullable=False, default="")
    # 结构化属性，如 {"材质": "...", "尺码": "...", "适用场景": "..."}
    details = db.Column(db.JSON, nullable=False, default=dict)
    # 使用 / 洗涤说明
    care_info = db.Column(db.Text, nullable=False, default="")
    # 发货 / 物流说明
    delivery_note = db.Column(db.Text, nullable=False, default="")
    # 退换说明
    return_note = db.Column(db.Text, nullable=False, default="")

    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def __repr__(self):
        return f"<Product {self.sku_id} {self.product_name}>"
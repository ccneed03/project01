"""
Role and Permission Models
"""
from datetime import datetime
from app.extensions import db


# Association tables
user_roles = db.Table(
    "user_roles",
    db.Column("user_id", db.Integer, db.ForeignKey("users.id"), primary_key=True),
    db.Column("role_id", db.Integer, db.ForeignKey("roles.id"), primary_key=True),
)

role_permissions = db.Table(
    "role_permissions",
    db.Column("role_id", db.Integer, db.ForeignKey("roles.id"), primary_key=True),
    db.Column("permission_id", db.Integer, db.ForeignKey("permissions.id"), primary_key=True),
)


class Permission(db.Model):
    """Permission model"""
    __tablename__ = "permissions"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    resource = db.Column(db.String(64), nullable=False, index=True)
    action = db.Column(db.String(32), nullable=False)
    description = db.Column(db.String(256))

    # Relationships
    roles = db.relationship("Role", secondary=role_permissions, back_populates="permissions")

    def __repr__(self):
        return f"<Permission {self.name}>"


class Role(db.Model):
    """Role model"""
    __tablename__ = "roles"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    description = db.Column(db.String(256))
    is_system = db.Column(db.Boolean, default=False)  # System roles cannot be deleted
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    users = db.relationship("User", secondary=user_roles, back_populates="roles")
    permissions = db.relationship("Permission", secondary=role_permissions, back_populates="roles")

    def has_permission(self, resource: str, action: str) -> bool:
        """Check if role has permission"""
        for perm in self.permissions:
            if perm.resource == resource and perm.action == action:
                return True
        return False

    def __repr__(self):
        return f"<Role {self.name}>"


# Default permissions
DEFAULT_PERMISSIONS = [
    # User permissions
    {"name": "user.read", "resource": "users", "action": "read", "description": "查看用户"},
    {"name": "user.write", "resource": "users", "action": "write", "description": "管理用户"},
    # Role permissions
    {"name": "role.read", "resource": "roles", "action": "read", "description": "查看角色"},
    {"name": "role.write", "resource": "roles", "action": "write", "description": "管理角色"},
    # Knowledge base permissions
    {"name": "kb.read", "resource": "knowledge_bases", "action": "read", "description": "查看知识库"},
    {"name": "kb.write", "resource": "knowledge_bases", "action": "write", "description": "管理知识库"},
    # Document permissions
    {"name": "doc.read", "resource": "documents", "action": "read", "description": "查看文档"},
    {"name": "doc.write", "resource": "documents", "action": "write", "description": "管理文档"},
    {"name": "doc.process", "resource": "documents", "action": "process", "description": "处理文档"},
    # Search permissions
    {"name": "search.read", "resource": "search", "action": "read", "description": "使用检索"},
    # Chat permissions
    {"name": "chat.read", "resource": "chat", "action": "read", "description": "查看对话"},
    {"name": "chat.write", "resource": "chat", "action": "write", "description": "管理对话"},
    # Demo after-sales permissions
    {"name": "after_sales.read", "resource": "after_sales", "action": "read", "description": "查询订单、物流和商品"},
    {"name": "after_sales.write", "resource": "after_sales", "action": "write", "description": "发起售后动作"},
]

# Default roles
DEFAULT_ROLES = [
    {
        "name": "superadmin",
        "description": "超级管理员，拥有所有权限",
        "is_system": True,
        "permissions": [p["name"] for p in DEFAULT_PERMISSIONS],
    },
    {
        "name": "admin",
        "description": "管理员，管理用户、知识库、文档",
        "is_system": True,
        "permissions": [
            "user.read", "user.write",
            "role.read",
            "kb.read", "kb.write",
            "doc.read", "doc.write", "doc.process",
            "search.read",
            "chat.read", "chat.write",
            "after_sales.read", "after_sales.write",
        ],
    },
    {
        "name": "user",
        "description": "普通用户，使用检索和问答",
        "is_system": True,
        "permissions": [
            "kb.read",
            "doc.read",
            "search.read",
            "chat.read", "chat.write",
            "after_sales.read", "after_sales.write",
        ],
    },
]

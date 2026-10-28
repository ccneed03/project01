"""
Authentication Service
"""
from app.models.user import User
from app.models.role import Role, Permission, DEFAULT_PERMISSIONS, DEFAULT_ROLES
from app.extensions import db


class AuthService:
    """Authentication service"""

    @staticmethod
    def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
        """Register a new user

        Returns:
            (success, message)
        """
        # Check if username exists
        if User.query.filter_by(username=username).first():
            return False, "用户名已存在"

        # Check if email exists
        if User.query.filter_by(email=email).first():
            return False, "邮箱已被注册"

        # Create user
        user = User(username=username, email=email)
        user.set_password(password)

        # Assign default role (user)
        default_role = Role.query.filter_by(name="user").first()
        if default_role:
            user.roles.append(default_role)

        db.session.add(user)
        db.session.commit()

        return True, "注册成功"

    @staticmethod
    def authenticate(username: str, password: str) -> tuple[bool, str, User]:
        """Authenticate user

        Returns:
            (success, message, user)
        """
        user = User.query.filter_by(username=username).first()

        if not user:
            return False, "用户名或密码错误", None

        if not user.check_password(password):
            return False, "用户名或密码错误", None

        if not user.is_active:
            return False, "账户已被禁用", None

        return True, "登录成功", user

    @staticmethod
    def init_default_data():
        """Initialize default roles and permissions"""
        # Create permissions
        for perm_data in DEFAULT_PERMISSIONS:
            if not Permission.query.filter_by(name=perm_data["name"]).first():
                perm = Permission(**perm_data)
                db.session.add(perm)

        db.session.commit()

        # Create roles
        for role_data in DEFAULT_ROLES:
            if not Role.query.filter_by(name=role_data["name"]).first():
                role = Role(
                    name=role_data["name"],
                    description=role_data["description"],
                    is_system=role_data["is_system"],
                )
                db.session.add(role)

                # Add permissions
                for perm_name in role_data["permissions"]:
                    perm = Permission.query.filter_by(name=perm_name).first()
                    if perm:
                        role.permissions.append(perm)

        db.session.commit()

        # Keep existing system roles in sync with newly added permissions.
        with db.session.no_autoflush:
            for role_data in DEFAULT_ROLES:
                role = Role.query.filter_by(name=role_data["name"]).first()
                if not role:
                    continue
                for perm_name in role_data["permissions"]:
                    perm = Permission.query.filter_by(name=perm_name).first()
                    if perm and perm not in role.permissions:
                        role.permissions.append(perm)
        db.session.commit()

        # Create superadmin if not exists
        if not User.query.filter_by(username="admin").first():
            admin = User(username="admin", email="admin@example.com", is_superuser=True)
            admin.set_password("admin123")
            superadmin_role = Role.query.filter_by(name="superadmin").first()
            if superadmin_role:
                admin.roles.append(superadmin_role)
            db.session.add(admin)
            db.session.commit()

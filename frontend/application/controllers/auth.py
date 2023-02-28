# Import Dependencies
import sqlalchemy

from flask import Blueprint, request, flash, redirect, url_for
from flask.templating import render_template
from flask_login import login_user
from flask_login.utils import login_required, logout_user
from werkzeug.security import check_password_hash

from .utils import *
from ..forms.sign_up_form import SignUpForm
from ..models.user import User


# Instantiate Blueprint
auth = Blueprint('auth', __name__)


# Login API
@auth.route('/auth/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    user = User.query.filter_by(username=username).first()

    if not (user and password and check_password_hash(user.password, password)):
        flash('Please check your login details and try again.', category='warning')
        return redirect(url_for('routes.login'))

    login_user(user=user, remember=True)
    return redirect(url_for('routes.home'))


# Sign up API
@auth.route('/auth/sign-up', methods=['POST'])
def sign_up():
    form = SignUpForm(request.form)
    if form.validate():
        try:
            email = request.form.get('email')
            new_username = request.form.get('new_username')
            new_password = request.form.get('new_password')
            new_user_id = add_new_user(email=email, username=new_username, password=new_password)
            new_user = get_user(new_user_id)
            login_user(user=new_user)
            return redirect(url_for('routes.home'))
        except sqlalchemy.exc.IntegrityError:
            flash('Email or Username has already been taken!', category='error')
            return redirect(url_for('routes.sign_up'))
    else:
        return render_template('sign-up.html', form=form, loginMode=False)


# Log out API
@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('routes.index'))

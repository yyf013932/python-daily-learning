from flask import Flask, make_response, request, url_for, redirect

app = Flask(__name__)


@app.route('/')
def hello_world():
    """
    最基本的访问方式
    :return:
    """
    return 'Hello World!'


@app.route('/user/<username>')
def show_user_profile(username):
    """
    可以解析url地址中的参数
    :param username: url地址中匹配<parameter>位置的内容将会作为参数解析
    """
    # show the user profile for that user
    return 'User %s' % username


@app.route("/resp_demo")
def resp_handle():
    resp = make_response("this is response demo")
    resp.headers['X-Something'] = 'A value'
    return resp


@app.route('/set_cookie/<username>')
def set_cookie(username):
    """
    [resp.set_cookie]
    设置cookie
    :param username:
    :return:
    """
    resp = make_response("cookie_demo")
    resp.set_cookie("name", username)
    return resp


@app.route('/get_cookie')
def get_cookie():
    """
    [request.cookies.get]
    获取cookie
    request是一个环境局部对象
    :return:
    """
    value = request.cookies.get("name")
    resp = make_response(value)
    return resp


@app.route('/url_para')
def url_parameter():
    """
    [request.args.get]
    获取url中的参数
    :return:
    """
    para = request.args.get('key', '')
    return "get key %s" % para


@app.route('/form_para')
def form_parameter():
    """
    [request.form]
    获取请求的表单参数
    :return:
    """
    username = request.form['username']
    password = request.form['password']
    return "get username = %s,password = %s" % (username, password)


@app.route('/redirect')
def redirect_demo():
    """
    [redirect]
    重定向
    :return:
    """
    return redirect("/")


@app.errorhandler(404)
def page_not_found(error):
    """
    自定义错误码处理
    :param error:
    :return:
    """
    return "this is 404 error"


app.run(debug=True)

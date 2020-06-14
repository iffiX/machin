try:
    import allure
except ImportError:
    def decorator_mocker(func):
        return func

    class AllureMocker(object):
        def __init__(self, root="allure"):
            self.root = root

        def __getattr__(self, item):
            query = self.root + "." + item
            if any([query.startswith(i) for i in (
                "allure.title",
                "allure.description",
                "allure.description_html",
                "allure.label",
                "allure.severity",
                "allure.epic",
                "allure.feature",
                "allure.story",
                "allure.suite",
                "allure.parent_suite",
                "allure.sub_suite",
                "allure.tag",
                "allure.id",
                "allure.link",
                "allure.issue",
                "allure.testcase"
            )]):
                return decorator_mocker
            return AllureMocker(self.root + "." + item)

        def __setattr__(self, key, item):
            pass

        def __call__(self, *args, **kwargs):
            return None

    allure = AllureMocker()

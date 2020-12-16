

class Tweet: 

    def __init__(self, id): 
        self._id = id
        self._user = None
        self._created_at = None
        self._text = None
        self._retweet_of = None
        self._retweeted_by = []

    @property
    def id(self):
        return self._id

    @property
    def created_at(self):
        return self._created_at

    @created_at.setter
    def created_at(self, value):
        self._created_at = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, value):
        self._user = value

    @property
    def is_retweet(self):
        return self._retweet_of is not None

    @property
    def retweet_of(self):
        return self._retweet_of

    @retweet_of.setter
    def retweet_of(self, value):
        self._retweet_of = value

    @property
    def retweeted_by(self):
        return self._retweeted_by

    def __repr__(self):
        return "Tweet(%r)" % self._id


class User: 

    def __init__(self, id):
        self._id = id
        self._followers = set()
        self._following = set()
        self._screenname = None
        self._botometer_scores = None

    @property
    def id(self):
        return self._id
    
    @property
    def followers(self):
        return self._followers

    @property
    def following(self) :
        return self._following

    @property
    def screenname(self):
        return self._screenname

    @screenname.setter
    def screenname(self, value):
        self._screenname = value

    @property
    def popularity(self):
        return len(self._followers)

    @property
    def botometer_scores(self):
        return self._botometer_scores

    @botometer_scores.setter
    def botometer_scores(self, value):
        self._botometer_scores = value

    def __repr__(self):
        return "User(%r)" % self._id
    

class BotometerScores: 

    def __init__(self):
        self.astroturf = 0.0
        self.fake_follower = 0.0
        self.financial = 0.0
        self.other = 0.0
        self.overall = 0.0
        self.self_declared = 0.0
        self.spammer = 0.0


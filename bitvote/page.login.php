<?php
require_once 'vendor/autoload.php';
use \Mailjet\Resources;

$UID=0;
$HS='';
if (isset($_SESSION['salt'])) 
   $HS=intval($_SESSION['salt']);
unset($_SESSION['salt']);
if (isset($_SESSION['UID']) && !isset($_REQUEST['email'])) 
   $UID=intval($_SESSION['UID']);
unset($_SESSION['UID']);
$sql='SELECT id FROM users WHERE id="'.$UID.'" AND passhash="'
   .hash('sha512',$salt[0].$HS.$salt[1].$_SERVER['REMOTE_ADDR']).'"';
$sql=SQL2ARRAY($sql);
if (count($sql)>0) {
   $_SESSION['UID']=$UID;
   $_SESSION['salt']=$HS;
}
unset($UID);
unset($HS);

if (!isset($_SESSION['UID']) && isset($_REQUEST['email'])) {
   $email=SQLIFY(strtolower($_REQUEST['email']));
   $valid=false;
   if ($email=='ceneezer@gmail.com')
      $valid=true;
   if (strpos($email, '@')>0 && (substr($email,-3)=='.ca' || substr($email,-4)=='.com'))
      $valid=true;
   if (!$valid) {
      if ($email!='') {
?>
<h4>We are currently only accepting email addresses ending in .ca and .com</h4>
<?php
      }
   } else {
      if (!isset($_REQUEST['salt'])) {      
//link from login form, should inclue only email
         $HS=rand(111111,9999999);
         $sql='INSERT INTO users (mpid, email, passhash) VALUES ("'.SQLIFY(gUIDv4(null, 'users', 'mpid')).'", "'
            .$email.'", "'
            .hash('sha512',$salt[0].$HS.$salt[1].$_SERVER['REMOTE_ADDR'])
            .'") ON DUPLICATE KEY UPDATE passhash="'
            .hash('sha512',$salt[0].$HS.$salt[1].$_SERVER['REMOTE_ADDR'])
            .'"';
MY_LOG('Attempted login by '.$email);
         SQL2ARRAY($sql);
         $sql='SELECT id FROM users WHERE email LIKE "'.$email.'" AND passhash="'
            .hash('sha512',$salt[0].$HS.$salt[1].$_SERVER['REMOTE_ADDR'])
            .'"';
//and datediff(stamp, now)... prob above
         $sql=SQL2ARRAY($sql);
         if (count($sql)>0) {
MY_LOG('Sending email: '.$email." ".$HS);

$mj = new \Mailjet\Client($MAILJET_KEY,$MAILJET_SECRET,true,['version' => 'v3.1']);
  $warn="Someone claiming to own this email requested login to bitvote - if it was not you simply ignore this message and they can proceed no further - if these are becoming spam please let us know at abuse@bitvote.ca and we will remove the ability for anyone (including you) to make such requests.\n\n";
  $body = [
    'Messages' => [
      [
        'From' => [
          'Email' => "noreply@bitvote.ca",
          'Name' => "BitVote Login"
        ],
        'To' => [
          [
            'Email' => "cto@digitizinghumanity.com"//,
//            'Name' => "Chris"
          ]
        ],
        'Subject' => "Login attempt from BitVote",
        'TextPart' => $warn.'To complete login, copy and paste this URL: https://'.$site.'?email='.$email.'&salt='.$HS,
        'HTMLPart' => $warn.'<br /><br />To complete login click this link: <a href="https://'.$site.'?email='.$email.'&salt='.$HS.'">Login</a>',
        'CustomID' => "login attempt"
      ]
    ]
  ];
  $response = $mj->post(Resources::$Email, ['body' => $body]);
//var_dump($response);
//MY_LOG('test'.serialize($response));

  if ($response->success()) {
MY_LOG('Mail sent: '.$email);
?>
<main><h2>Please check your email (including spam folder) for your single use login link - it may take a few minutes to arrive but is on it's way.</h2></main></body></html>
<?php 
die();
  } else {
MY_LOG('Mail failed: '.$email);
  }
         }
      } else {
//link from email, should include $_REQUEST['salt'] and email
         $HS=SQLIFY($_REQUEST['salt']);
         $sql='SELECT id FROM users WHERE email LIKE "'.$email.'" AND passhash="'
            .hash('sha512',$salt[0].$HS.$salt[1].$_SERVER['REMOTE_ADDR']).'"';
         $sql=SQL2ARRAY($sql);
         if (count($sql)>0) {
            $_SESSION['UID']=$sql[0][0];
            $HS=rand(111111,9999999);
            $sql='INSERT INTO users (email) VALUES ("'
               .$email.'") ON DUPLICATE KEY UPDATE passhash="'
               .hash('sha512',$salt[0].$HS.$salt[1].$_SERVER['REMOTE_ADDR'])
               .'"';
            SQL2ARRAY($sql);
            $_SESSION['salt']=$HS;
MY_LOG('Login succesful!');
         }
//redirect to remove querry
?><main>
<h3>If you are not automatically redirected click <a href="">here</a></h3>
<script>document.location.href='?page=issues';</script></main></body></html>
<?php
die();
      }
   }
} 
if (!isset($_SESSION['UID'])) { ?>
<form class="login" method="post">
   <input type="submit" value="Login Link" />
   <input type="email" name="email" placeholder="Email" />
</form>
<?php } else { ?>
<form class="logout" method="post">
   <input type="hidden" name="email" value="" />
   <input type="submit" value="Logout" />
</form>
<?php } ?>
